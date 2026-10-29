"""
Node-matching check without MD (CPU only): track_nodes with effective_phonons stubbed
to return the harmonic force constants. Every Γ/R node must keep its own bands and
charge, although node 7's remembered frequency is moved onto bands 4-7, a trap for a
frequency-only matcher (the bug fixed in c5f712f).
    python smoke_track_nodes.py ../POSCAR ../runs/gamma_R_vs_T_fixed/phonopy_params.yaml
"""
import sys, types, phonopy
from ase.io import read
from vasp_grace.condmat import topological_phonons as tp
from vasp_grace.condmat.phonon_tools import PhonopyPhonons, eigensystem

atoms = read(sys.argv[1]); ph = PhonopyPhonons(phonopy.load(sys.argv[2]))
ca = dict(radius=0.01, n_theta=16, tol=1e-3)
base = []
for q in ([0, 0, 0], [.5, .5, .5]):
    nu, _ = eigensystem(ph, q)
    for lo, hi in tp.multiplets(nu, 1e-3):
        if hi > lo and nu[lo] > 0.1:
            n = tp.node_chern(ph, atoms, q, lo, hi, **ca); n["fixed"] = True; base.append(n)
next(n for n in base if tuple(n["bands"]) == (0, 3))["frequency_THz"] = 5.26
tp.effective_phonons = lambda *a, **k: {"ph": ph, "halves": (ph, ph), "prim": atoms,
    "info": {"temperature_K": a[2], "volume": "stub", "volume_per_cell_A3": atoms.get_volume(),
             "space_group": "stub", "space_group_kept": True, "n_snapshots": 0, "sample_every_fs": 0,
             "rms_displacement_A": 0, "fit_residual": 0, "mean_force_per_atom_eV_A": 0,
             "static_offset": 0}}
args = types.SimpleNamespace(supercell=[3, 3, 3], md_timestep=2, npt_steps=0, md_equil=0,
                             md_snapshots=0, md_every=0, no_npt=False, seed=0, degen_tol=1e-3)
rows, _, events, _ = tp.track_nodes(base, [100], atoms, None, args, ca, ph)
ok = True
for r in (r for r in rows if r["T"] == 100):
    h = next(x for x in rows if x["T"] == 0 and x["id"] == r["id"])
    good = tuple(r["bands"]) == tuple(h["bands"]) and r["charges"] == h["charges"] and r["status"] == "tracked"
    ok &= good
    print(r["id"], r["bands"], round(r["freq"], 3), r["charges"], r["status"], "" if good else "  <-- FAIL")
print("PASS" if ok and not events else f"FAIL {events}")
