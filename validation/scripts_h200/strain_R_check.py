"""
R-node charge of CoSi under uniaxial strain along z, on spheres of several radii,
and the Weyl points between bands 1 and 2 (inside the fourfold R multiplet) within
0.1 1/Å of R. Run from validation/strain_z_tension_cosi (CPU is enough):
    python ../scripts_h200/strain_R_check.py
"""
import types, numpy as np
from scipy.optimize import minimize
from ase.io import read
from vasp_grace.calculator import load_calculator
from vasp_grace.condmat import topological_phonons as tp
from vasp_grace.condmat.phonon_tools import eigensystem

prim = read("POSCAR"); calc = load_calculator("GRACE-2L-OAM", verbose=False)
args = types.SimpleNamespace(supercell=[3, 3, 3], delta=0.015, strain_axis="z")
stage = tp.strain_stage(prim, calc, args)
R, rng = np.array([0.5, 0.5, 0.5]), np.random.default_rng(1)
for e in (0.005, 0.01):
    eff = stage(e); ph, a = eff["ph"], eff["prim"]
    rec = np.asarray(a.cell.reciprocal()); inv = np.linalg.inv(rec)
    dist = lambda q: np.linalg.norm(((q - R + 0.5) % 1 - 0.5) @ rec)
    print(f"strain {e}: R frequencies {np.round(eigensystem(ph, R)[0][:8], 4)}")
    for r in (0.0025, 0.005, 0.01, 0.02):
        n = tp.node_chern(ph, a, R, 0, 3, radius=r, tol=1e-3)
        print(f"  R, bands 0-3, r = {r}: charges {n['charges']}")
    gap = lambda q: (lambda nu: nu[2] - nu[1])(eigensystem(ph, q)[0])
    found = []
    for _ in range(300):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        res = minimize(gap, R + (d * rng.uniform(0.005, 0.08)) @ inv, method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 800})
        if res.fun < 2e-4 and dist(res.x) > 2e-3 and not any(
                np.linalg.norm(((res.x - f + 0.5) % 1 - 0.5) @ rec) < 1e-3 for f, _ in found):
            n = tp.node_chern(ph, a, res.x, 1, 2, radius=min(0.0025, dist(res.x) / 3), tol=1e-4)
            found.append((res.x, n["gap_charge"].get(1)))
    near = sorted((dist(q), c) for q, c in found if dist(q) < 0.1)
    print(f"  gap-1|2 nodes within 0.1 1/Å of R: {len(near)}, (distance 1/Å, charge): "
          + ", ".join(f"({d:.4f}, {c})" for d, c in near), flush=True)
