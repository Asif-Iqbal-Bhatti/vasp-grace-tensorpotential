#!/usr/bin/env python3
"""
topological_phonons.py

Topological analysis of phonon band structures computed with GRACE MLIP.

Detects band crossings (potential Weyl/Dirac phonon points), computes the
local Berry curvature around each crossing, and classifies phononic topology.
Builds on the existing phonon module (main.py, IBRION=5/6) — reads cached
force constant data (phonon.*.json files).

Physics background
------------------
The phonon band structure can carry non-trivial topological invariants:
    - Weyl phonons   : point-like band crossings with non-zero Chern number
                       (chiral phonons, topological surface arcs)
    - Nodal-line phonons : line of band crossings forming a loop in k-space
    - Dirac phonons  : fourfold-degenerate crossing (two Weyl of opposite chirality)

A point node is classified by its topological charge, the Chern number of the
bands below it on a small sphere around the node (--chern, --node, --search):
    - C = ±1          → Weyl phonon
    - C = ±2          → double-Weyl, or a threefold spin-1 node (bands +2, 0, -2)
    - C = 0           → no monopole; with a Berry phase γ = π on a loop, a nodal line
The Berry phase on a loop alone (--berry) is π both for a nodal line and for a
loop centred on a Weyl point, so it cannot separate them.
Over the whole Brillouin zone the charges in each gap sum to zero
(Nielsen–Ninomiya); --search reports this balance as a completeness check.

Usage
-----
    # Uses cached phonon data (phonon.*.json) from main.py
    grace-topo-phonons --poscar POSCAR --model GRACE-2L-OAM

    # Custom k-path and crossing threshold
    grace-topo-phonons --poscar POSCAR --model GRACE-2L-OAM \\
                                  --path GXMGZ --nkpts 300 --tol 0.1

    # Berry phase and Chern number at each detected crossing
    grace-topo-phonons --poscar POSCAR --model GRACE-2L-OAM --berry --chern

    # Charges of every degeneracy at Γ and R (e.g. spin-1 and fourfold nodes in CoSi)
    grace-topo-phonons --poscar POSCAR --node 0 0 0 --node 0.5 0.5 0.5

    # Search the whole zone for nodes in gaps 10..14 and check the charge balance
    grace-topo-phonons --poscar POSCAR --search 12 --bands 10 14

Outputs
-------
    phonon_topo_bands.png     band structure with crossings highlighted
    phonon_crossings.dat      table of detected crossings (k, bands, gap, Berry phase)
    phonon_nodes.dat          nodes with band Chern numbers and classification
                              (with --chern, --node or --search)
    phonon_topo_summary.txt   summary of topological classification
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ase.io import read
from ase.phonons import Phonons

try:
    from ..calculator import load_calculator
    from .phonon_tools import (bands_and_vectors, eigensystem, chern_on_sphere, find_nodes,
                               have_phonopy, phonopy_phonons, effective_phonons)
except ImportError:  # run as a plain script; needs vasp_grace installed
    from vasp_grace.calculator import load_calculator
    from vasp_grace.condmat.phonon_tools import (bands_and_vectors, eigensystem,
                                                 chern_on_sphere, find_nodes,
                                                 have_phonopy, phonopy_phonons,
                                                 effective_phonons)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Phonon setup
# ──────────────────────────────────────────────────────────────────────────────

def build_phonon_object(atoms, calc, supercell=(3, 3, 3), delta=0.015):
    """
    Initialise ASE Phonons from cached displacement data (phonon.*.json).
    If no cached data is found, runs fresh finite-displacement calculations.
    """
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta, name="phonon")

    cache_exists = os.path.exists(f"phonon.{0}.{0}.json")
    if cache_exists:
        print("Loading cached phonon force constants (phonon.*.json)...")
        ph.read(method="Frederiksen", symmetrize=3, acoustic=True,
                cutoff=None, born=False)
    else:
        print("No cached phonon data found. Running finite-displacement calculations...")
        ph.run()
        ph.read(method="Frederiksen", symmetrize=3, acoustic=True,
                cutoff=None, born=False)
    return ph


# ──────────────────────────────────────────────────────────────────────────────
# Band structure + eigenvectors
# ──────────────────────────────────────────────────────────────────────────────

def get_bands_and_modes(ph, atoms, path_str, n_kpts):
    """
    Compute phonon frequencies and eigenvectors along a k-path.

    Returns
    -------
    q_pts    : (N, 3) k-points in fractional coordinates
    omega_kn : (N, n_modes) frequencies in THz
    u_kn     : (N, n_modes, n_modes) orthonormal eigenvectors
               u_kn[k][n, :] = eigenvector for band n at k-point k
    x_coords : (N,) x-axis coordinates for plotting
    special_x: list of (x, label) for high-symmetry points
    """
    path = atoms.cell.bandpath(path_str, npoints=n_kpts)
    q_pts = path.kpts

    print(f"Computing phonon eigenvectors at {len(q_pts)} k-points...")
    # THz frequencies and orthonormal eigenvectors of the dynamical matrix
    omega_kn, u_kn = bands_and_vectors(ph, q_pts)

    # Build x-axis from path
    x_coords, special_x = _build_x_axis(path, q_pts, atoms)

    return q_pts, omega_kn, u_kn, x_coords, special_x


def _build_x_axis(path, q_pts, atoms):
    """Build cumulative distance x-axis for band structure plot."""
    rec = np.array(atoms.cell.reciprocal())
    q_cart = q_pts @ rec
    dq = np.linalg.norm(np.diff(q_cart, axis=0), axis=1)
    x = np.concatenate([[0], np.cumsum(dq)])

    # Special points from path
    special_x = []
    if hasattr(path, 'special_points') and path.special_points:
        # Try to get high-symmetry labels from path object
        try:
            for label, kpt in path.special_points.items():
                kpt_cart = np.array(kpt) @ rec
                dists = np.linalg.norm(q_cart - kpt_cart[None, :], axis=1)
                idx = np.argmin(dists)
                special_x.append((x[idx], label))
        except Exception:
            pass

    return x, special_x


# ──────────────────────────────────────────────────────────────────────────────
# Band crossing detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_crossings(omega_kn, x_coords, tol_THz=0.1):
    """
    Detect band crossings: adjacent bands approaching within `tol_THz` THz.

    Two complementary criteria:
    1. Near-degeneracy: |ω_n(k) - ω_{n+1}(k)| < tol
    2. Band inversion: gap changes sign between adjacent k-points

    Returns list of dicts with keys: k_idx, x, band_lo, band_hi, gap, type
    """
    n_kpts, n_modes = omega_kn.shape
    crossings = []
    seen = set()

    for k in range(1, n_kpts - 1):
        for n in range(n_modes - 1):
            gap_k   = omega_kn[k,     n + 1] - omega_kn[k,     n]
            gap_km1 = omega_kn[k - 1, n + 1] - omega_kn[k - 1, n]

            # Near-degeneracy at this k
            near_degen = abs(gap_k) < tol_THz

            # Sign change in gap between k-1 and k (band inversion)
            inversion = gap_k * gap_km1 < 0

            if near_degen or inversion:
                key = (k, n)
                if key not in seen:
                    seen.add(key)
                    crossing_type = "inversion" if inversion else "degeneracy"
                    crossings.append({
                        "k_idx"   : k,
                        "x"       : x_coords[k],
                        "band_lo" : n,
                        "band_hi" : n + 1,
                        "gap"     : abs(gap_k),
                        "type"    : crossing_type,
                        "berry"   : None,     # filled later if --berry
                    })

    # Remove duplicates (same crossing detected at multiple k shifts)
    merged = _merge_nearby_crossings(crossings, x_coords)
    return merged


def _merge_nearby_crossings(crossings, x_coords, merge_window=3):
    """Merge crossings within `merge_window` k-points of each other."""
    if not crossings:
        return []
    merged, used = [], set()
    for i, c in enumerate(crossings):
        if i in used:
            continue
        group = [c]
        for j, c2 in enumerate(crossings[i+1:], start=i+1):
            if j not in used and c2["band_lo"] == c["band_lo"]:
                if abs(c2["k_idx"] - c["k_idx"]) <= merge_window:
                    group.append(c2)
                    used.add(j)
        # Keep the one with smallest gap
        best = min(group, key=lambda g: g["gap"])
        merged.append(best)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Local Berry phase around a crossing
# ──────────────────────────────────────────────────────────────────────────────

def compute_berry_phase_at_crossing(ph, atoms, q0_frac, band_n, n_loop=20, radius=0.02):
    """
    Compute Berry phase γ for phonon band `band_n` on a small closed loop
    in the qx-qy plane centred at q0_frac.

    γ = 0   → trivial (no topological charge)
    γ = π   → nodal line or Weyl point; the loop alone cannot tell them
              apart, use node_chern() for the charge

    Returns the Berry phase in radians.
    """
    # Build a small circular loop in reduced coordinates
    angles = np.linspace(0, 2 * np.pi, n_loop, endpoint=False)
    loop_kpts = np.array([
        q0_frac + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    ])
    # Close the loop
    loop_kpts = np.vstack([loop_kpts, loop_kpts[0]])

    _, u_loop = bands_and_vectors(ph, loop_kpts)

    # Wilson loop product
    product = 1.0 + 0j
    n_pts = len(u_loop) - 1
    for j in range(n_pts):
        uj  = u_loop[j][band_n, :]
        uj1 = u_loop[j + 1][band_n, :]
        overlap = np.vdot(uj, uj1)
        if abs(overlap) > 1e-10:
            product *= overlap / abs(overlap)

    gamma = -np.imag(np.log(product))
    return gamma


# ──────────────────────────────────────────────────────────────────────────────
# Chern number on a sphere around a node
# ──────────────────────────────────────────────────────────────────────────────

def multiplets(nu, tol=0.05):
    """Group bands at one q into multiplets: consecutive gaps below tol THz chain together."""
    groups, lo = [], 0
    for n in range(1, len(nu) + 1):
        if n == len(nu) or nu[n] - nu[n - 1] >= tol:
            groups.append((lo, n - 1))
            lo = n
    return groups


def degenerate_multiplet(nu, band, tol=0.05):
    """The multiplet (lo, hi) that contains `band` at one q."""
    return next(g for g in multiplets(nu, tol) if g[0] <= band <= g[1])


def same_node(a, b, rec, tol):
    """
    Same multiplet within `tol` (1/Å, no 2π) modulo a reciprocal lattice vector.
    Finite-displacement force constants are not exactly space-group symmetric,
    so symmetry partners found independently sit slightly apart; nodes closer
    than the sphere radius cannot be told apart by the Chern calculation anyway.
    """
    d = (np.asarray(a["q"]) - np.asarray(b["q"]) + 0.5) % 1.0 - 0.5
    return a["bands"] == b["bands"] and np.linalg.norm(d @ rec) < tol


def symmetry_images(node, rotations):
    """
    Copies of a node at all symmetry-equivalent q, with their charges.

    `rotations` are spglib's integer rotations W (fractional direct-lattice
    basis). A reduced q transforms as q' = W^-T q; a proper rotation keeps the
    charge and an improper one flips it. Time reversal maps q -> -q and keeps
    a phonon Weyl charge.
    """
    out = []
    for W in rotations:
        sign = int(round(np.linalg.det(W)))
        Wq = np.linalg.inv(W).T
        for tr in (1, -1):
            img = dict(node)
            img["q"] = np.mod(tr * (Wq @ node["q"]), 1.0)
            img["charges"] = {g: sign * c for g, c in node["charges"].items()}
            img["band_chern"] = [None if c is None else sign * c for c in node["band_chern"]]
            img["from_symmetry"] = True
            out.append(img)
    return out


def unique_nodes(nodes, rec, tol):
    out = []
    for n in nodes:
        if not any(same_node(n, m, rec, tol) for m in out):
            out.append(n)
    return out


def node_chern(ph, atoms, q0, band_lo, band_hi, radius=0.01, n_theta=16, tol=0.05,
               gap_floor=1e-3):
    """
    Topological charge of the crossing between bands band_lo..band_hi at q0.

    The node's bands are widened to the full degenerate multiplet lo..hi at q0
    (a threefold node involves three bands). On a sphere of `radius` (1/Å, no
    2π) around q0, the node's charge in each internal gap g (lo <= g < hi) is
    C(bands <= g) - C(bands <= lo-1); the subtraction removes other nodes at
    the same q (e.g. lower multiplets at Γ).

    A gap counts only if its Chern number is the same integer on spheres of
    radius r and r/2 and the gap stays above `gap_floor` THz on both. Bands can
    remain paired along lines or planes through the node (screw axes), and Weyl
    points can split very slowly, so a fixed gap cut would either accept the
    first or reject the second; agreement between two radii separates them.
    """
    nu0, _ = eigensystem(ph, q0)
    lo, _ = degenerate_multiplet(nu0, band_lo, tol)
    _, hi = degenerate_multiplet(nu0, band_hi, tol)
    eig = lambda q: eigensystem(ph, q)
    rec = atoms.cell.reciprocal()
    res = chern_on_sphere(eig, q0, radius, rec, (lo, hi), n_theta=n_theta)
    half = chern_on_sphere(eig, q0, radius / 2, rec, (lo, hi), n_theta=n_theta)
    is_open = lambda g: g < 0 or (
        min(res["min_gap"].get(g, 0.0), half["min_gap"].get(g, 0.0)) > gap_floor
        and res["gap_charge"].get(g) == half["gap_charge"].get(g))
    base = res["gap_charge"].get(lo - 1, 0)
    charges = {g: res["gap_charge"][g] - base
               for g in range(lo, hi) if is_open(g) and is_open(lo - 1)}
    band_chern = [res["band_chern"][m] if (m in res["band_chern"] and is_open(m)
                                           and is_open(m - 1)) else None
                  for m in range(lo, hi + 1)]
    return {
        "q": np.asarray(q0, float),
        "bands": (lo, hi),
        "multiplicity": hi - lo + 1,
        "frequency_THz": float(np.mean(nu0[lo:hi + 1])),
        "charges": charges,
        "band_chern": band_chern,
        "gap_charge": {g: c - base for g, c in res["gap_charge"].items() if g >= lo},
        "open_gaps": [g for g in range(lo - 1, hi + 1) if g >= 0 and is_open(g)],
        "unresolved_gaps": [g for g in range(lo, hi) if g not in charges],
    }


def classify_node(node, berry=None):
    """Name the node from its charges in the open internal gaps and its multiplicity."""
    if not node["charges"]:
        return ("unresolved: charge differs between r and r/2 or the gap closes on the "
                "sphere (nodal line or surface, or a second node within --sphere_r)")
    cmax = max(abs(c) for c in node["charges"].values())
    m = node["multiplicity"]
    if cmax == 0:
        if berry is not None and abs(abs(berry) % (2 * np.pi) - np.pi) < 0.3:
            return "C = 0, Berry phase π: nodal line"
        return "C = 0: no monopole charge"
    if m == 2:
        return {1: "Weyl point", 2: "double-Weyl point"}.get(cmax, f"charge-{cmax} node")
    if m == 3:
        return f"threefold (spin-1) node, |C| = {cmax}"
    if m == 4:
        return (f"fourfold node, charge {cmax} between its band pairs"
                + (" (charge-2 Dirac)" if cmax == 2 else ""))
    return f"{m}-fold node, |C| = {cmax}"


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_crossing(berry_phase):
    """Classify based on Berry phase (radians)."""
    if berry_phase is None:
        return "unknown"
    gamma_mod = abs(berry_phase) % (2 * np.pi)
    if gamma_mod < 0.3 or gamma_mod > 2 * np.pi - 0.3:
        return "trivial"
    elif abs(gamma_mod - np.pi) < 0.3:
        return "nodal line or Weyl (γ≈π)"
    else:
        return f"non-trivial (γ={np.degrees(berry_phase):.1f}°)"


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def plot_bands_with_crossings(x_coords, omega_kn, crossings, special_x,
                               filename="phonon_topo_bands.png"):
    n_modes = omega_kn.shape[1]
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot all bands
    for n in range(n_modes):
        ax.plot(x_coords, omega_kn[:, n], "b-", lw=0.8, alpha=0.6)

    # Highlight crossings
    colors = {"inversion": "red", "degeneracy": "orange"}
    labels = {"inversion": "Band inversion", "degeneracy": "Near-degeneracy"}
    plotted = set()
    for c in crossings:
        ctype = c["type"]
        lbl = labels[ctype] if ctype not in plotted else None
        ax.axvline(c["x"], color=colors[ctype], lw=1.2, alpha=0.7, ls="--", label=lbl)
        n_lo, n_hi = c["band_lo"], c["band_hi"]
        y_mid = 0.5 * (omega_kn[c["k_idx"], n_lo] + omega_kn[c["k_idx"], n_hi])
        berry = c.get("berry")
        marker = "★" if berry is not None and abs(abs(berry) % (2*np.pi) - np.pi) < 0.3 else "×"
        ax.text(c["x"], y_mid, marker, ha="center", va="center",
                fontsize=10, color=colors[ctype])
        plotted.add(ctype)

    # High-symmetry labels
    for x_sp, label in special_x:
        ax.axvline(x_sp, color="k", lw=0.5, alpha=0.4)
        ax.text(x_sp, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1,
                label, ha="center", va="top", fontsize=9)

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_xlabel("Wave vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Phonon band structure — topological analysis")
    if plotted:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


def write_crossings(crossings, q_pts, filename="phonon_crossings.dat"):
    with open(filename, "w") as f:
        f.write("# Phonon band crossings detected\n")
        f.write(f"{'#idx':>5}  {'qx':>8}  {'qy':>8}  {'qz':>8}  "
                f"{'band_lo':>8}  {'band_hi':>8}  {'gap(THz)':>10}  "
                f"{'type':>12}  {'berry(rad)':>12}  {'class':>25}\n")
        for i, c in enumerate(crossings):
            q = q_pts[c["k_idx"]]
            berry = c["berry"] if c["berry"] is not None else float("nan")
            cls = classify_crossing(c["berry"])
            f.write(f"{i:5d}  {q[0]:8.5f}  {q[1]:8.5f}  {q[2]:8.5f}  "
                    f"{c['band_lo']:8d}  {c['band_hi']:8d}  {c['gap']:10.4f}  "
                    f"{c['type']:>12}  {berry:12.5f}  {cls:>25}\n")
    print(f"Wrote {filename}  ({len(crossings)} crossings)")


def write_nodes(nodes, filename="phonon_nodes.dat"):
    with open(filename, "w") as f:
        f.write("# Phonon nodes with Chern numbers on a small sphere (A = i<u|grad u>, outward normal)\n")
        f.write("# charges: node charge in each internal gap g (bands <= g) that is open on the sphere\n")
        f.write("# band Chern: per band, '?' where a neighbouring gap closes on the sphere\n")
        f.write(f"# {'qx':>8}  {'qy':>8}  {'qz':>8}  {'bands':>9}  {'mult':>4}  {'freq(THz)':>10}  "
                f"{'charges':>18}  {'band Chern':>16}  class\n")
        for n in nodes:
            q = n["q"]
            bc = ",".join("?" if c is None else f"{c:+d}" for c in n["band_chern"])
            ch = ",".join(f"{g}:{c:+d}" for g, c in n["charges"].items()) or "-"
            f.write(f"  {q[0]:8.5f}  {q[1]:8.5f}  {q[2]:8.5f}  {n['bands'][0]:>4}-{n['bands'][1]:<4}  "
                    f"{n['multiplicity']:4d}  {n['frequency_THz']:10.4f}  {ch:>18}  {bc:>16}  "
                    f"{n['class']}\n")
    print(f"Wrote {filename}  ({len(nodes)} nodes)")


def charge_balance(nodes):
    """
    Total charge per gap over the nodes found, and the number of unresolved
    crossings in that gap. A gap's total is zero when its point nodes are all
    found; unresolved crossings (nodal lines or surfaces) can carry the rest.
    """
    total, unresolved = {}, {}
    for n in nodes:
        for gap, c in n["charges"].items():
            total[gap] = total.get(gap, 0) + c
        for gap in n["unresolved_gaps"]:
            unresolved[gap] = unresolved.get(gap, 0) + 1
    return {g: (total.get(g, 0), unresolved.get(g, 0))
            for g in sorted(set(total) | set(unresolved))
            if total.get(g, 0) != 0 or unresolved.get(g, 0)}


def write_summary(crossings, n_modes, n_imaginary, filename="phonon_topo_summary.txt",
                  nodes=None, balance=None):
    n_weyl = sum(1 for c in crossings
                 if c["berry"] is not None
                 and abs(abs(c["berry"]) % (2*np.pi) - np.pi) < 0.3)
    with open(filename, "w") as f:
        f.write("Topological phonon analysis summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total phonon modes      : {n_modes}\n")
        f.write(f"Imaginary modes (< 0)   : {n_imaginary}\n")
        f.write(f"Band crossings detected : {len(crossings)}\n")
        f.write(f"  - by band inversion   : {sum(1 for c in crossings if c['type']=='inversion')}\n")
        f.write(f"  - by near-degeneracy  : {sum(1 for c in crossings if c['type']=='degeneracy')}\n")
        f.write(f"Crossings with γ ≈ π    : {n_weyl}  (nodal line or Weyl; see Chern numbers)\n")
        if nodes:
            f.write(f"\nNodes with Chern numbers: {len(nodes)} (phonon_nodes.dat)\n")
            for n in nodes:
                f.write(f"  q = ({n['q'][0]:.4f}, {n['q'][1]:.4f}, {n['q'][2]:.4f})  "
                        f"bands {n['bands'][0]}-{n['bands'][1]}  "
                        f"charges {n['charges']}  {n['class']}\n")
        if balance is not None:
            f.write("\nCharge balance over the zone (Nielsen-Ninomiya)\n")
            closed = [g for g, (c, u) in balance.items() if c == 0]
            if not any(c for c, _ in balance.values()):
                f.write("  every gap with point nodes sums to zero\n")
            for g, (c, u) in balance.items():
                if c == 0 and not u:
                    continue
                note = ("closes" if c == 0 else
                        "open: point nodes missed, or charge on unresolved lines/surfaces"
                        if u else "open: some point nodes were not found")
                f.write(f"  gap {g:3d}|{g + 1:<3d}  total {c:+d}   unresolved crossings {u:3d}   {note}\n")
            if closed:
                f.write(f"  gaps that close with unresolved crossings present: {closed}\n")
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# Phonon topology versus temperature
# ──────────────────────────────────────────────────────────────────────────────

def _multiplet_spread(ph, q, lo, hi):
    """Frequency spread of bands lo..hi at q: zero exactly at the node."""
    nu = eigensystem(ph, q)[0]
    return float(nu[hi] - nu[lo])


def follow_node(ph, q0, lo, hi, fixed=False):
    """
    Re-converge a node from its previous position with new force constants.
    Nodes pinned by symmetry (e.g. at Γ or R, given with --node) stay where they
    are. Returns (q, spread in THz); spread above a tolerance means the node is gone.
    """
    from scipy.optimize import minimize
    q0 = np.asarray(q0, float)
    if fixed:
        return q0, _multiplet_spread(ph, q0, lo, hi)
    res = minimize(lambda q: _multiplet_spread(ph, q, lo, hi), q0, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-8, "maxiter": 600,
                            "initial_simplex": q0 + np.vstack([np.zeros(3), np.eye(3) * 0.01])})
    return np.mod(res.x, 1.0), float(res.fun)


def _min_image_distance(qa, qb, rec):
    d = (np.asarray(qa) - np.asarray(qb) + 0.5) % 1.0 - 0.5
    return float(np.linalg.norm(d @ rec))


def _assign_pinned(nodes, nu, vecs, degen_tol):
    """
    One-to-one assignment of symmetry-pinned nodes at one q to the multiplets of
    the new spectrum there. Bands reorder as frequencies shift with T, so each
    node is matched by the overlap of its multiplet with the previous one,
    |V_old† V_new|²/m (1 for the same modes, unchanged by mixing inside the
    multiplet), frequency difference breaking ties, with the Hungarian algorithm.
    Returns {node id: ((lo, hi), overlap)}, or None for a node left without a
    multiplet of its size.
    """
    from scipy.optimize import linear_sum_assignment
    groups = [g for g in multiplets(nu, degen_tol) if g[1] > g[0] and nu[g[0]] > 0.1]
    overlap = np.zeros((len(nodes), len(groups)))
    cost = np.full_like(overlap, 1e6)
    for i, t in enumerate(nodes):
        for j, (lo, hi) in enumerate(groups):
            if hi - lo + 1 == t["mult"]:
                overlap[i, j] = np.linalg.norm(t["vecs"].conj().T @ vecs[:, lo:hi + 1]) ** 2 / t["mult"]
                cost[i, j] = 1.0 - overlap[i, j] + 0.01 * abs(nu[lo:hi + 1].mean() - t["freq"])
    match = {t["id"]: None for t in nodes}
    for i, j in zip(*linear_sum_assignment(cost)):
        if cost[i, j] < 1e6:
            match[nodes[i]["id"]] = (groups[j], float(overlap[i, j]))
    return match


def track_nodes(base_nodes, temperatures, prim, calc, args, chern_args, ph0, node_tol=0.02,
                min_overlap=0.5, lost_overlap=0.1):
    """
    Follow every node from the harmonic (0 K) result through increasing T.

    At each T the effective force constants are fitted from GRACE MD; each node
    is re-converged from its previous position, and its charge recomputed. The
    two half-fits give error bars on the node position and frequency. A node
    whose bands no longer meet has vanished; two vanishing nodes of opposite
    charge in the same gap that were close together have annihilated. Nodes
    pinned by symmetry are matched one-to-one to the multiplets at their q by
    eigenvector overlap with the previous T (`ph0` gives the 0 K eigenvectors);
    an overlap below `min_overlap` is reported, as the modes have mixed, and one
    below `lost_overlap` (or no multiplet of the node's size left) loses the node.
    """
    rec0 = np.asarray(prim.cell.reciprocal())
    tracked = [{"id": i, "bands": n["bands"], "q": np.asarray(n["q"], float),
                "charges": dict(n["charges"]), "fixed": n.get("fixed", False), "alive": True,
                "freq": n["frequency_THz"], "mult": n["multiplicity"]}
               for i, n in enumerate(base_nodes) if n["charges"]]
    for t in tracked:
        if t["fixed"]:
            t["vecs"] = eigensystem(ph0, t["q"])[1][:, t["bands"][0]:t["bands"][1] + 1]
    rows = [{"T": 0.0, "id": t["id"], "bands": t["bands"], "q": t["q"],
             "freq": base_nodes[t["id"]]["frequency_THz"], "err_freq": 0.0, "err_q": 0.0,
             "charges": t["charges"], "status": "harmonic"} for t in tracked]
    md_info, events, last = [], [], None

    for T in sorted(temperatures):
        print(f"\nT = {T:g} K: GRACE MD and effective force constants...")
        eff = effective_phonons(prim, calc, T, tuple(args.supercell), timestep=args.md_timestep,
                                npt_steps=args.npt_steps, equil_steps=args.md_equil,
                                n_snapshots=args.md_snapshots, sample_every=args.md_every,
                                npt=not args.no_npt, seed=args.seed)
        info = eff["info"]
        md_info.append(info)
        print(f"  {info['volume']}: V = {info['volume_per_cell_A3']:.3f} Å³, "
              f"{info['space_group']}, fit residual {info['fit_residual']:.3f}, "
              f"mean force {info['mean_force_per_atom_eV_A']:.4f} eV/Å")
        rec = np.asarray(eff["prim"].cell.reciprocal())
        ph_a, ph_b = eff["halves"]
        nodes_T, gone = [], []
        # nodes pinned by symmetry keep their q and multiplicity but can change band
        # index when frequencies reorder: assign them per q, one node per multiplet
        match, spectra = {}, {}
        for qkey in {tuple(np.round(t["q"], 6)) for t in tracked if t["fixed"] and t["alive"]}:
            at_q = [t for t in tracked if t["fixed"] and t["alive"]
                    and tuple(np.round(t["q"], 6)) == qkey]
            spectra[qkey] = eigensystem(eff["ph"], at_q[0]["q"])
            match.update(_assign_pinned(at_q, *spectra[qkey], args.degen_tol))
        for t in tracked:
            if not t["alive"]:
                continue
            note = ""
            if t["fixed"]:
                bands, ov = match[t["id"]] or (None, 0.0)
                if ov < lost_overlap:
                    t["alive"] = False
                    why = (f"no free {t['mult']}-fold multiplet at its q" if bands is None else
                           f"best free multiplet {bands[0]}-{bands[1]} has overlap {ov:.2f}")
                    rows.append({"T": float(T), "id": t["id"], "bands": t["bands"], "q": t["q"],
                                 "freq": t["freq"], "err_freq": 0.0, "err_q": 0.0, "charges": {},
                                 "status": f"lost ({why})"})
                    events.append(f"T = {T:g} K: node {t['id']} (bands {t['bands'][0]}-"
                                  f"{t['bands'][1]}) lost: {why}")
                    continue
                t["bands"] = bands
                t["vecs"] = spectra[tuple(np.round(t["q"], 6))][1][:, t["bands"][0]:t["bands"][1] + 1]
                if ov < min_overlap:
                    note = f", modes mixed (overlap {ov:.2f})"
            lo, hi = t["bands"]
            q, spread = follow_node(eff["ph"], t["q"], lo, hi, t["fixed"])
            qa, _ = follow_node(ph_a, t["q"], lo, hi, t["fixed"])
            qb, _ = follow_node(ph_b, t["q"], lo, hi, t["fixed"])
            nu, nua, nub = (eigensystem(x, y)[0] for x, y in ((eff["ph"], q), (ph_a, qa), (ph_b, qb)))
            row = {"T": float(T), "id": t["id"], "bands": t["bands"], "q": q,
                   "freq": float(nu[lo:hi + 1].mean()),
                   "err_freq": float(abs(nua[lo:hi + 1].mean() - nub[lo:hi + 1].mean())),
                   "err_q": _min_image_distance(qa, qb, rec), "charges": {}, "status": ""}
            if spread > node_tol:
                t["alive"] = False
                row["status"] = f"vanished (bands {spread:.3f} THz apart)"
                gone.append((t, t["q"]))
            else:
                node = node_chern(eff["ph"], eff["prim"], q, lo, hi, **chern_args)
                node["class"] = classify_node(node)
                node["frequency_THz"] = row["freq"]
                nodes_T.append(node)
                row["charges"] = node["charges"]
                same_charges = sorted(node["charges"].values()) == sorted(t["charges"].values())
                row["status"] = ("tracked" if same_charges
                                 else f"charges changed from {t['charges']}") + note
                t["q"], t["freq"] = q, row["freq"]
            rows.append(row)
        # opposite charges vanishing together close by: pair annihilation
        for i, (a, qa_) in enumerate(gone):
            for b, qb_ in gone[i + 1:]:
                ga, gb = set(a["charges"]), set(b["charges"])
                common = ga & gb
                if common and all(a["charges"][g] == -b["charges"][g] for g in common) \
                        and _min_image_distance(qa_, qb_, rec0) < 0.05:
                    events.append(f"T = {T:g} K: nodes {a['id']} and {b['id']} "
                                  f"(gap {sorted(common)}) annihilated")
        for t, _ in gone:
            events.append(f"T = {T:g} K: node {t['id']} (bands {t['bands'][0]}-{t['bands'][1]}) vanished")
        write_nodes(nodes_T, f"phonon_nodes_T{T:g}.dat")
        last = (T, eff)
    return rows, md_info, events, last


def find_new_nodes(eff, T, rows, gaps, args, chern_args):
    """Full search at the highest T for nodes the tracking did not start from."""
    print(f"\nSearching the zone at T = {T:g} K for newly created nodes...")
    ph, prim = eff["ph"], eff["prim"]
    rec = np.asarray(prim.cell.reciprocal())
    known = [r for r in rows if r["T"] == T and r["charges"]]
    new = []
    for f in find_nodes(lambda q: eigensystem(ph, q), args.search, gaps, tol_THz=args.tol,
                        max_nodes=args.max_seeds):
        if any(_min_image_distance(f["q"], r["q"], rec) < args.sphere_r and
               r["bands"][0] <= f["band_lo"] <= r["bands"][1] for r in known):
            continue
        node = node_chern(ph, prim, f["q"], f["band_lo"], f["band_lo"] + 1, **chern_args)
        if node["charges"] and not any(same_node(node, m, rec, args.sphere_r) for m in new):
            node["class"] = classify_node(node)
            new.append(node)
    return new


def write_trajectories(rows, md_info, events, new_nodes, filename="weyl_trajectories.dat",
                       summary="phonon_topo_vs_T.txt"):
    with open(filename, "w") as f:
        f.write("# Phonon nodes versus temperature (effective force constants from GRACE MD)\n")
        f.write("# err_freq, err_q: difference between fits to the two halves of the MD run\n")
        f.write(f"# {'T(K)':>6}  {'id':>4}  {'bands':>7}  {'qx':>8}  {'qy':>8}  {'qz':>8}  "
                f"{'freq(THz)':>9}  {'err_freq':>8}  {'err_q(1/Å)':>10}  {'charges':>14}  status\n")
        for r in rows:
            ch = ",".join(f"{g}:{c:+d}" for g, c in r["charges"].items()) or "-"
            f.write(f"  {r['T']:6.0f}  {r['id']:4d}  {r['bands'][0]:>3}-{r['bands'][1]:<3}  "
                    f"{r['q'][0]:8.5f}  {r['q'][1]:8.5f}  {r['q'][2]:8.5f}  {r['freq']:9.4f}  "
                    f"{r['err_freq']:8.4f}  {r['err_q']:10.5f}  {ch:>14}  {r['status']}\n")
    with open(summary, "w") as f:
        f.write("Phonon topology versus temperature\n==================================\n\n")
        f.write("Effective harmonic force constants at T, fitted by phonopy (symfc) from classical\n"
                "GRACE MD snapshots. No zero-point motion. Error bars: two half-run fits.\n\n")
        for i in md_info:
            f.write(f"T = {i['temperature_K']:g} K: {i['volume']}, V = {i['volume_per_cell_A3']:.3f} Å³, "
                    f"{i['space_group']}{'' if i['space_group_kept'] else ' (SPACE GROUP CHANGED)'}, "
                    f"{i['n_snapshots']} snapshots every {i['sample_every_fs']:g} fs, "
                    f"rms displacement {i['rms_displacement_A']:.3f} Å, fit residual "
                    f"{i['fit_residual']:.3f}, mean force {i['mean_force_per_atom_eV_A']:.4f} eV/Å\n")
        f.write("\nEvents\n------\n")
        for e in events or ["none: every node survived"]:
            f.write(f"- {e}\n")
        if new_nodes is not None:
            f.write(f"\nNodes found only at the highest T: {len(new_nodes)}\n")
            for n in new_nodes:
                f.write(f"  q = ({n['q'][0]:.4f}, {n['q'][1]:.4f}, {n['q'][2]:.4f})  bands "
                        f"{n['bands'][0]}-{n['bands'][1]}  charges {n['charges']}\n")
    print(f"Wrote {filename} and {summary}")


def plot_trajectories(rows, filename="weyl_nodes_vs_T.png"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for nid in sorted({r["id"] for r in rows}):
        rs = [r for r in rows if r["id"] == nid and (r["charges"] or r["status"] == "harmonic")]
        if len(rs) < 2:
            continue
        sign = next(iter(rs[0]["charges"].values()), 0)
        ax.errorbar([r["T"] for r in rs], [r["freq"] for r in rs],
                    yerr=[r["err_freq"] for r in rs], marker="o", ms=3, lw=1, capsize=2,
                    color="tab:red" if sign > 0 else "tab:blue", alpha=0.7)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Node frequency (THz)")
    ax.set_title("Phonon nodes versus temperature (red C > 0, blue C < 0)")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Topological phonon analysis: detect Weyl/Dirac phonon points with GRACE."
    )
    parser.add_argument("--poscar",  default="POSCAR",       help="Unit cell POSCAR.")
    parser.add_argument("--model",   default="GRACE-2L-OAM", help="GRACE model.")
    parser.add_argument("--path",    default="GXMGZ",        help="k-path string (ASE notation, default: GXMGZ).")
    parser.add_argument("--nkpts",   type=int, default=200,  help="Number of k-points (default: 200).")
    parser.add_argument("--supercell", type=int, nargs=3, default=[3,3,3],
                        metavar=("Nx","Ny","Nz"), help="Phonon supercell (default: 3 3 3).")
    parser.add_argument("--delta",   type=float, default=0.015,
                        help="Displacement amplitude in Å (default: 0.015).")
    parser.add_argument("--fc", choices=["phonopy", "ase"], default=None,
                        help="Force constants: phonopy (space-group symmetrized, the default when "
                             "phonopy is installed) or ase (finite displacements, cached in phonon/).")
    parser.add_argument("--tol",     type=float, default=0.1,
                        help="Crossing detection threshold in THz (default: 0.1).")
    parser.add_argument("--berry",   action="store_true",
                        help="Compute Berry phase at each detected crossing (slower).")
    parser.add_argument("--loop_pts",type=int, default=30,
                        help="k-points on Berry phase loop (default: 30).")
    parser.add_argument("--loop_r",  type=float, default=0.02,
                        help="Berry loop radius in reduced coords (default: 0.02).")
    parser.add_argument("--chern",   action="store_true",
                        help="Chern number on a small sphere around each crossing on the path.")
    parser.add_argument("--node",    type=float, nargs=3, action="append", metavar=("qx", "qy", "qz"),
                        help="Also analyse every degeneracy at this reduced q (repeatable), e.g. --node 0 0 0.")
    parser.add_argument("--search",  type=int, metavar="N",
                        help="Search the whole zone on an N×N×N mesh for nodes and compute their charges.")
    parser.add_argument("--bands",   type=int, nargs=2, metavar=("LO", "HI"),
                        help="Restrict --search to gaps LO..HI (default: all optical gaps).")
    parser.add_argument("--max_seeds", type=int, default=60,
                        help="Crossings refined per gap in --search (default: 60).")
    parser.add_argument("--sphere_r", type=float, default=0.01,
                        help="Sphere radius in 1/Å without 2π (default: 0.01).")
    parser.add_argument("--sphere_n", type=int, default=16,
                        help="Polar divisions of the sphere mesh (default: 16).")
    parser.add_argument("--degen_tol", type=float, default=None,
                        help="Bands within this many THz at a node form one multiplet (default: 0.001 "
                             "with phonopy, whose degeneracies are exact; 0.05 with --fc ase).")
    t_grp = parser.add_argument_group("phonon topology versus temperature (EXPERIMENTAL, see TODO.md; "
                                      "needs phonopy with symfc)")
    t_grp.add_argument("--temperature", type=float, nargs="+", metavar="T",
                       help="Follow the nodes from 0 K through these temperatures (K), using effective "
                            "force constants fitted from GRACE MD.")
    t_grp.add_argument("--md_snapshots", type=int, default=60, help="MD snapshots per T (default: 60).")
    t_grp.add_argument("--md_every", type=int, default=100,
                       help="MD steps between snapshots (default: 100, i.e. 200 fs).")
    t_grp.add_argument("--md_equil", type=int, default=2000, help="NVT equilibration steps (default: 2000).")
    t_grp.add_argument("--npt_steps", type=int, default=3000,
                       help="NPT steps for the average cell at T (default: 3000).")
    t_grp.add_argument("--no_npt", action="store_true",
                       help="Keep the 0 K cell (anharmonic shift at fixed volume only).")
    t_grp.add_argument("--md_timestep", type=float, default=2.0, help="MD time step in fs (default: 2).")
    t_grp.add_argument("--no_search_tmax", action="store_true",
                       help="With --search, skip the extra zone search at the highest T.")
    t_grp.add_argument("--seed", type=int, default=0, help="Random seed for the MD (default: 0).")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    atoms = read(args.poscar, format="vasp")
    calc  = load_calculator(args.model)

    fc = args.fc or ("phonopy" if have_phonopy() else "ase")
    if args.temperature and fc != "phonopy":
        parser.error("--temperature needs phonopy (for the symfc fit); install it or drop --fc ase")
    if args.degen_tol is None:
        args.degen_tol = 1e-3 if fc == "phonopy" else 0.05
    if fc == "phonopy":
        ph = phonopy_phonons(atoms, calc, tuple(args.supercell), args.delta)
    else:
        ph = build_phonon_object(atoms, calc, tuple(args.supercell), args.delta)
    print(f"Force constants: {fc}")

    q_pts, omega_kn, u_kn, x_coords, special_x = get_bands_and_modes(
        ph, atoms, args.path, args.nkpts
    )

    n_imaginary = int(np.sum(omega_kn < 0))
    n_modes = omega_kn.shape[1]
    print(f"Modes: {n_modes}  |  Imaginary: {n_imaginary}")

    crossings = detect_crossings(omega_kn, x_coords, tol_THz=args.tol)
    print(f"Detected {len(crossings)} crossing(s) with threshold {args.tol} THz")

    # Optional Berry phase computation
    if args.berry and u_kn is not None:
        print(f"\nComputing Berry phase at each crossing (loop_pts={args.loop_pts}, r={args.loop_r})...")
        for i, c in enumerate(crossings):
            q0 = q_pts[c["k_idx"]]
            gamma = compute_berry_phase_at_crossing(
                ph, atoms, q0, c["band_lo"],
                n_loop=args.loop_pts, radius=args.loop_r
            )
            c["berry"] = gamma
            cls = classify_crossing(gamma)
            print(f"  Crossing {i}: bands {c['band_lo']}-{c['band_hi']}  "
                  f"γ = {np.degrees(gamma):.1f}°  → {cls}" if gamma is not None
                  else f"  Crossing {i}: Berry phase failed")

    # Chern numbers: crossings on the path, user-given q-points, zone-wide search
    nodes, balance = [], None
    rec = np.asarray(atoms.cell.reciprocal())
    chern_args = dict(radius=args.sphere_r, n_theta=args.sphere_n, tol=args.degen_tol)
    if args.chern:
        print(f"\nChern numbers on spheres of radius {args.sphere_r} 1/Å...")
        for c in crossings:
            node = node_chern(ph, atoms, q_pts[c["k_idx"]], c["band_lo"], c["band_hi"], **chern_args)
            node["class"] = classify_node(node, c.get("berry"))
            c["chern"] = node["band_chern"]
            nodes.append(node)
    point_nodes = []
    for q in args.node or []:
        nu0, _ = eigensystem(ph, q)
        for lo, hi in multiplets(nu0, args.degen_tol):
            if hi > lo and nu0[lo] > 0.1:             # skip the acoustic zero at Γ
                node = node_chern(ph, atoms, q, lo, hi, **chern_args)
                node["class"] = classify_node(node)
                node["fixed"] = True                  # pinned to this q for --temperature
                point_nodes.append(node)
    base_nodes = list(point_nodes)
    gaps = None
    if args.search:
        n_modes_all = omega_kn.shape[1]
        gaps = range(args.bands[0], args.bands[1] + 1) if args.bands else range(3, n_modes_all - 1)
        print(f"\nSearching the zone on a {args.search}³ mesh for nodes in gaps "
              f"{gaps.start}..{gaps.stop - 1}...")
        found = find_nodes(lambda q: eigensystem(ph, q), args.search, gaps, tol_THz=args.tol,
                           max_nodes=args.max_seeds)
        print(f"  {len(found)} crossing point(s) found")
        searched = []
        for f in found:
            node = node_chern(ph, atoms, f["q"], f["band_lo"], f["band_lo"] + 1, **chern_args)
            node["class"] = classify_node(node)
            searched.append(node)
        # complete each node's symmetry orbit, then keep one entry per multiplet and q
        import spglib
        ds = spglib.get_symmetry_dataset(
            (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=1e-3)
        rotations = ds.rotations if ds is not None else [np.eye(3, dtype=int)]
        computed = unique_nodes(point_nodes + searched, rec, args.sphere_r)
        base_nodes = computed
        point_nodes = unique_nodes(computed + [img for n in computed
                                               for img in symmetry_images(n, rotations)],
                                   rec, args.sphere_r)
        print(f"  {len(computed)} distinct nodes computed, {len(point_nodes)} after adding "
              f"symmetry-equivalent copies ({len(rotations)} rotations × time reversal)")
        balance = {g: v for g, v in charge_balance(point_nodes).items() if g in gaps}
    nodes = unique_nodes(nodes + point_nodes, rec, args.sphere_r)
    for n in nodes:
        print(f"  q = ({n['q'][0]:.4f}, {n['q'][1]:.4f}, {n['q'][2]:.4f})  bands {n['bands'][0]}-"
              f"{n['bands'][1]}  ν = {n['frequency_THz']:.3f} THz  charges {n['charges']}  → {n['class']}")
    if nodes:
        write_nodes(nodes)

    plot_bands_with_crossings(x_coords, omega_kn, crossings, special_x)
    write_crossings(crossings, q_pts)
    write_summary(crossings, n_modes, n_imaginary, nodes=nodes, balance=balance)

    if args.temperature:
        if not base_nodes:
            parser.error("--temperature needs nodes to follow: give --node and/or --search")
        rows, md_info, events, (t_max, eff_max) = track_nodes(
            base_nodes, args.temperature, atoms, calc, args, chern_args, ph)
        new_nodes = None
        if args.search and not args.no_search_tmax:
            new_nodes = find_new_nodes(eff_max, t_max, rows, gaps, args, chern_args)
        write_trajectories(rows, md_info, events, new_nodes)
        plot_trajectories(rows)


if __name__ == "__main__":
    main()
