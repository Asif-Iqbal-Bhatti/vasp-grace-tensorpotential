"""
Offline diagnostics on MD snapshots saved by md_dump.py (CPU only, no GRACE needed):
refits with other references, Γ-mode projections, the static offset and the cell
mismatch between the MD and the reference.
    python refit.py ../runs/gamma_diagnosis/md_100K_before_fix.npz \
                    ../runs/gamma_R_vs_T_fixed/phonopy_params.yaml
The second argument is the harmonic (0 K) phonopy file used as the reference.
"""
import sys, json, numpy as np, phonopy
from ase import Atoms, units
from vasp_grace.condmat import phonon_tools as pt
d = np.load(sys.argv[1])
info = json.loads(str(d["info"])); T = info["temperature_K"]
prim = Atoms(numbers=d["numbers"], cell=d["prim_cell"], scaled_positions=d["prim_scaled"], pbc=True)
prim_ref = Atoms(numbers=d["numbers"], cell=d["ref_cell"], scaled_positions=d["ref_scaled"], pbc=True)
U, F = d["U"], d["F"]
SC = (3, 3, 3)
harm = pt.PhonopyPhonons(phonopy.load(sys.argv[2]))
def gam(ph): return pt.eigensystem(ph, [0, 0, 0])[0][[6, 9, 12, 17, 21]]
print("info", info)
print("harmonic Γ", np.round(gam(harm), 3))
ph, r = pt._fit_force_constants(prim_ref, SC, U, F); print("refit as run Γ", np.round(gam(ph), 3), f"res {r:.4f}")

# harmonic force constants on the MD data
fc = harm.phonon.force_constants
if fc.shape[0] != fc.shape[1]:
    from phonopy.harmonic.force_constants import compact_fc_to_full_fc
    fc = compact_fc_to_full_fc(harm.phonon.primitive, fc)
pred = -np.einsum("ijab,sjb->sia", fc, U)
print(f"harmonic FC residual on MD data {np.linalg.norm(F - pred)/np.linalg.norm(F):.4f}")
print("mean force per atom (eV/Å)", np.linalg.norm(F.mean(0), axis=1).mean(),
      " |mean U| per atom (Å)", np.linalg.norm(U.mean(0), axis=1).mean())

# intercept: centre U and F over snapshots
ph, r = pt._fit_force_constants(prim_ref, SC, U - U.mean(0), F - F.mean(0)); print("centred fit Γ", np.round(gam(ph), 3), f"res {r:.4f}")

# reference = 0 K Wyckoff positions in the T cell
p0 = prim.copy(); p0.set_cell(prim_ref.cell, scale_atoms=True)
sc_ref = pt._ase_supercell(pt._phonopy_for(prim_ref, SC)).get_positions()
sc_0 = pt._ase_supercell(pt._phonopy_for(p0, SC)).get_positions()
U0 = U + (sc_ref - sc_0); U0 -= U0.mean(axis=1, keepdims=True)
ph, r = pt._fit_force_constants(p0, SC, U0, F); print("ref = 0 K Wyckoff Γ", np.round(gam(ph), 3), f"res {r:.4f}")
print("Wyckoff shift ref vs 0 K (Å):", np.abs(prim_ref.positions - p0.positions - (prim_ref.positions - p0.positions).mean(0)).max())

# Γ-mode projections: equipartition ω² = kT/<Q²>, force-displacement ω² = -<F_Q Q>/<Q²>
php = harm.phonon
masses = np.array(pt._phonopy_for(prim, SC).supercell.masses)
s2u = pt._phonopy_for(prim, SC).supercell.s2u_map; u2u = pt._phonopy_for(prim, SC).supercell.u2u_map
to_prim = np.array([u2u[i] for i in s2u]); nc = len(masses) // len(prim)
nu, vecs = pt.eigensystem(harm, [0, 0, 0])
conv = pt._SQRT_TO_EV * pt.THZ_PER_EV
kT = units.kB * T
for band in (6, 9, 12, 17, 21):
    e = vecs[:, band].conj().reshape(-1, 3)[to_prim]                     # Γ pattern on supercell
    Q = np.einsum("ia,sia->s", e * np.sqrt(masses)[:, None], U) / np.sqrt(nc)
    FQ = np.einsum("ia,sia->s", e / np.sqrt(masses)[:, None], F) / np.sqrt(nc)
    Q2 = np.mean(abs(Q) ** 2); w_eq = np.sqrt(kT / Q2) * conv
    w_fd = np.sqrt(max(-np.mean((FQ.conj() * Q).real) / Q2, 0)) * conv
    w_ff = (np.mean(abs(FQ) ** 2) / kT) ** 0.5 * conv                       # <F_Q²> = ω² kT
    print(f"band {band}: harmonic {nu[band]:.3f}  equipartition {w_eq:.3f}  <FQ·Q> {w_fd:.3f}  <F²> {w_ff:.3f}  |<Q>| {abs(Q.mean()):.4f} rms {np.sqrt(Q2):.4f}")

# triplet as a whole: static offset relative to thermal rms in the degenerate subspace
for lo in (6, 9, 12, 17, 21):
    Qs = []
    for band in range(lo, lo + 3):
        e = vecs[:, band].conj().reshape(-1, 3)[to_prim]
        Qs.append(np.einsum("ia,sia->s", e * np.sqrt(masses)[:, None], U) / np.sqrt(nc))
    Qs = np.array(Qs)
    print(f"triplet {lo}-{lo+2}: |<Q>|/rms = {np.linalg.norm(Qs.mean(1)) / np.sqrt(np.mean(abs(Qs)**2) * 3):.3f}")

# cell mismatch: mean U linear in the reference fractional coordinates means U contains (h_A - h_B) s
frac = pt._ase_supercell(pt._phonopy_for(prim_ref, SC)).get_scaled_positions()
A = np.hstack([frac, np.ones((len(frac), 1))])
coef, *_ = np.linalg.lstsq(A, U.mean(0), rcond=None)
sc_cell = np.asarray(pt._ase_supercell(pt._phonopy_for(prim_ref, SC)).cell)
print("mean-U slope (Å per unit supercell fraction):\n", np.round(coef[:3], 5))
print("implied strain (slope / supercell length):\n", np.round(coef[:3] / np.linalg.norm(sc_cell, axis=1)[:, None], 6))
print("reference cell (symmetrized):\n", np.asarray(prim_ref.cell))
