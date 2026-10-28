"""
phonon_tools.py

Shared phonon eigen-solver and lattice Berry-phase tools for the topology modules.

Eigenvectors
------------
ASE's Phonons.band_structure(modes=True) returns frequencies in eV and mode
patterns scaled by 1/√m. Those patterns are not orthonormal, so they are the
wrong objects for Berry phases. Here the mass-weighted dynamical matrix D(q)
is diagonalized directly: its eigenvectors are orthonormal, and frequencies are
returned in THz (negative values mark imaginary modes).

Chern numbers
-------------
Fukui, Hatsugai and Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005). On a mesh over
a closed surface, the link variable between neighbouring points for a set of
bands is U = det⟨u(k)|u(k')⟩ / |det|, and the Berry flux through a plaquette is
F = -arg(U₁ U₂ U₃ U₄). The flux sum is 2π × an integer, exactly, as soon as the
mesh is fine enough, and it is gauge invariant, so eigenvectors from separate
diagonalizations can be used as they come. Convention: Berry connection
A = i⟨u|∇u⟩, outward normal. With it the lower band of H = k·σ has C = +1.

On a small sphere around a band crossing, the Chern number of the bands below
a gap is the topological charge enclosed in that gap: ±1 for a Weyl point, ±2
for a double-Weyl or spin-1 node, 0 for a crossing that is not a monopole
(e.g. a point on a nodal line). Over the whole Brillouin zone the charges in
each gap sum to zero (Nielsen–Ninomiya), which checks a node search for
completeness.
"""

import numpy as np
from ase import units

THZ_PER_EV = units._e / units._hplanck / 1e12
# sqrt(eV / Å² / amu) -> eV, the factor ASE uses for phonon energies
_SQRT_TO_EV = units._hbar * 1e10 / np.sqrt(units._e * units._amu)


# ──────────────────────────────────────────────────────────────────────────────
# Force constants with space-group symmetry (phonopy)
# ──────────────────────────────────────────────────────────────────────────────

class PhonopyPhonons:
    """
    Force constants from phonopy, symmetrized over the space group, behind the
    interface eigensystem() uses.

    Finite-displacement force constants from ASE are not exactly space-group
    symmetric, so symmetry-related phonon nodes come out slightly displaced from
    one another. phonopy builds the force constants from symmetry-reduced
    displacements and projects them onto the space group, which removes that.

    phonopy puts atomic positions into the Fourier phase, so its D(q) is not
    periodic in q. dynamical_matrix() returns U† D U with U = diag(exp(-2πi q·r)),
    the lattice-periodic gauge that ASE uses (D(q+G) = D(q)), so meshes can wrap
    around the zone. The change of gauge is a diagonal phase; it leaves the
    frequencies untouched and does not change Chern numbers on closed surfaces.

    Every D(q) comes from phonopy's own get_dynamical_matrix_at_q().
    """

    def __init__(self, phonon):
        self.phonon = phonon
        self._pos = np.repeat(phonon.primitive.scaled_positions, 3, axis=0)

    def dynamical_matrix(self, q_red):
        q = np.asarray(q_red, dtype=float)
        D = self.phonon.get_dynamical_matrix_at_q(q)
        u = np.exp(-2j * np.pi * (self._pos @ q))
        return (u.conj()[:, None] * D) * u[None, :]


def have_phonopy():
    import importlib.util
    return importlib.util.find_spec("phonopy") is not None


def phonopy_phonons(atoms, calc, supercell=(3, 3, 3), delta=0.01, save="phonopy_params.yaml"):
    """
    Space-group-symmetrized force constants for `atoms` (the primitive cell)
    with forces from `calc`. Writes phonopy_params.yaml for reuse in phonopy.
    """
    from ase import Atoms
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())
    phonon = Phonopy(unit, supercell_matrix=np.diag(supercell))
    phonon.generate_displacements(distance=delta)
    forces = []
    for sc in phonon.supercells_with_displacements:
        a = Atoms(sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        a.calc = calc
        forces.append(a.get_forces())
    print(f"phonopy: {len(forces)} symmetry-reduced displacements in a "
          f"{'×'.join(map(str, supercell))} supercell")
    phonon.forces = np.array(forces)
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants(show_drift=False)
    phonon.symmetrize_force_constants_by_space_group(show_drift=False)
    if save:
        phonon.save(save)
    return PhonopyPhonons(phonon)


# ──────────────────────────────────────────────────────────────────────────────
# Eigen-solver
# ──────────────────────────────────────────────────────────────────────────────

def eigensystem(ph, q_red):
    """
    Frequencies (THz, ascending; negative = imaginary) and orthonormal
    eigenvectors (columns, shape (3N, 3N)) of the dynamical matrix at reduced q.
    `ph` is an ASE Phonons object after ph.read(), or a PhonopyPhonons.
    """
    if hasattr(ph, "dynamical_matrix"):
        D = ph.dynamical_matrix(q_red)
    else:
        # ASE's D(q) corresponds to Bloch waves exp(-iq·R); conjugate it to the
        # standard exp(+iq·R) convention phonopy uses, so both backends give the
        # same sign of Berry curvature and Chern numbers.
        D = ph.compute_dynamical_matrix(np.asarray(q_red, dtype=float), ph.D_N).conj()
    D = 0.5 * (D + D.conj().T)
    w2, vecs = np.linalg.eigh(D)
    nu = np.sign(w2) * np.sqrt(np.abs(w2)) * _SQRT_TO_EV * THZ_PER_EV
    return nu, vecs


def bands_and_vectors(ph, qs):
    """Frequencies (nq, nb) in THz and eigenvectors (nq, nb, 3N), band-major rows."""
    nus, vecs = zip(*(eigensystem(ph, q) for q in qs))
    return np.array(nus), np.transpose(np.array(vecs), (0, 2, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Lattice Chern numbers
# ──────────────────────────────────────────────────────────────────────────────

def _plaquette_flux(vec_grid, bands, wrap_i):
    """
    Total Berry flux / 2π of the band set `bands` over a grid of eigenvectors.

    vec_grid[i][j] : (3N, nb) eigenvectors at mesh point (i, j)
    wrap_i         : True for a torus (periodic in i), False for a sphere grid
                     whose first and last rows are the poles.
    j is always periodic.
    """
    ni, nj = len(vec_grid), len(vec_grid[0])
    b = list(bands)

    def link(a, c):
        d = np.linalg.det(a[:, b].conj().T @ c[:, b])
        return d / abs(d) if abs(d) > 1e-14 else 1.0

    total = 0.0
    for i in range(ni if wrap_i else ni - 1):
        i2 = (i + 1) % ni
        for j in range(nj):
            j2 = (j + 1) % nj
            u = (link(vec_grid[i][j], vec_grid[i2][j])
                 * link(vec_grid[i2][j], vec_grid[i2][j2])
                 * link(vec_grid[i2][j2], vec_grid[i][j2])
                 * link(vec_grid[i][j2], vec_grid[i][j]))
            total -= np.angle(u)
    return total / (2 * np.pi)


def sphere_points(center_red, radius, rec, n_theta=16, n_phi=None):
    """
    Reduced q-points on a sphere of `radius` (1/Å, no 2π) around center_red.
    Returns an (n_theta+1) × n_phi nested list; rows 0 and n_theta are the poles.
    """
    n_phi = n_phi or 2 * n_theta
    rec = np.asarray(rec)
    inv_rec = np.linalg.inv(rec)
    c_cart = np.asarray(center_red, float) @ rec
    grid = []
    for t in np.linspace(0.0, np.pi, n_theta + 1):
        row = []
        for p in np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False):
            n = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            row.append((c_cart + radius * n) @ inv_rec)
        grid.append(row)
    return grid


def chern_on_sphere(eig, center_red, radius, rec, bands_window, n_theta=16, n_phi=None):
    """
    Chern numbers on a small sphere around a band crossing.

    eig(q_red) -> (nu_THz, vecs)   e.g. lambda q: eigensystem(ph, q)
    bands_window = (lo, hi)        band indices to report (inclusive)

    Returns dict:
        gap_charge[n]  Chern number of bands 0..n, i.e. the charge in gap n|n+1,
                       for n = lo-1 .. hi
        band_chern[m]  Chern number of band m = gap_charge[m] - gap_charge[m-1]
        min_gap[n]     smallest frequency gap n|n+1 on the sphere (THz);
                       the charge of gap n is only meaningful when this is open
    """
    grid = sphere_points(center_red, radius, rec, n_theta, n_phi)
    # each pole is one k-point: compute it once and reuse it around the ring
    data = []
    for i, row in enumerate(grid):
        if i in (0, len(grid) - 1):
            e = eig(row[0])
            data.append([e] * len(row))
        else:
            data.append([eig(q) for q in row])
    vec_grid = [[d[1] for d in row] for row in data]
    nus = np.array([d[0] for row in data for d in row])

    lo, hi = bands_window
    gap_charge, min_gap = {}, {}
    for n in range(max(lo - 1, 0), hi + 1):
        if n >= nus.shape[1] - 1:
            break
        gap_charge[n] = int(round(_plaquette_flux(vec_grid, range(n + 1), wrap_i=False)))
        min_gap[n] = float((nus[:, n + 1] - nus[:, n]).min())
    if lo == 0:
        gap_charge[-1] = 0
    if hi >= nus.shape[1] - 1:
        gap_charge[nus.shape[1] - 1] = 0          # all bands together carry no charge
    band_chern = {m: gap_charge[m] - gap_charge[m - 1]
                  for m in range(lo, hi + 1) if m in gap_charge and m - 1 in gap_charge}
    return {"gap_charge": gap_charge, "band_chern": band_chern, "min_gap": min_gap}


def chern_on_torus(vec_grid, bands):
    """Chern number of `bands` on a periodic 2D mesh (e.g. a k_z = const plane)."""
    return _plaquette_flux(vec_grid, bands, wrap_i=True)


def plaquette_flux_map(vec_grid, band):
    """Per-plaquette Berry flux (radians) of one band on a periodic 2D mesh."""
    ni, nj = len(vec_grid), len(vec_grid[0])

    def link(x, y):
        z = np.vdot(x[:, band], y[:, band])
        return z / abs(z) if abs(z) > 1e-14 else 1.0

    out = np.zeros((ni, nj))
    for i in range(ni):
        for j in range(nj):
            a, b = vec_grid[i][j], vec_grid[(i + 1) % ni][j]
            c, d = vec_grid[(i + 1) % ni][(j + 1) % nj], vec_grid[i][(j + 1) % nj]
            out[i, j] = -np.angle(link(a, b) * link(b, c) * link(c, d) * link(d, a))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Node search over the whole Brillouin zone
# ──────────────────────────────────────────────────────────────────────────────

def find_nodes(eig, n_grid, bands, tol_THz=0.3, node_tol_THz=0.02, max_nodes=60):
    """
    Locate band crossings (gap n|n+1 -> 0) anywhere in the Brillouin zone.

    Scans an n_grid³ mesh, keeps points where gap n|n+1 is a local minimum below
    tol_THz, refines each by Nelder–Mead on the gap and keeps those that close
    to below node_tol_THz. Returns a list of dicts (q, band_lo, gap) with
    duplicates (same q modulo reciprocal lattice) merged.
    """
    from scipy.optimize import minimize

    axis = np.arange(n_grid) / n_grid
    Q = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    nus = np.array([eig(q)[0] for q in Q]).reshape(n_grid, n_grid, n_grid, -1)

    nodes = []
    for n in bands:
        gap = nus[..., n + 1] - nus[..., n]
        neighbours = [np.roll(gap, s, axis=a) for a in range(3) for s in (1, -1)]
        is_min = (gap < tol_THz) & np.all([gap <= g for g in neighbours], axis=0)
        seeds = np.argwhere(is_min)
        seeds = seeds[np.argsort(gap[tuple(seeds.T)])][:max_nodes]
        for idx in seeds:
            q0 = idx / n_grid
            f = lambda q, n=n: (lambda nu: nu[n + 1] - nu[n])(eig(q)[0])
            res = minimize(f, q0, method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-7, "maxiter": 400,
                                    "initial_simplex": q0 + np.vstack(
                                        [np.zeros(3), np.eye(3) * 0.5 / n_grid])})
            if res.fun > node_tol_THz:
                continue
            q = np.mod(res.x, 1.0)
            dup = any(n == m["band_lo"] and
                      np.linalg.norm((q - m["q"] + 0.5) % 1.0 - 0.5) < 2e-3
                      for m in nodes)
            if not dup:
                nodes.append({"q": q, "band_lo": n, "gap": float(res.fun)})
    return nodes
