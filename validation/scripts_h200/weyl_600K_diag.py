"""
CPU-only check of the Weyl nodes whose charge changed at 600 K in validation/weyl_vs_T_h200
(ids 45-77), from the effective force constants saved by fc_dump.py. Run from
validation/weyl_600K_diagnosis:
    python ../scripts_h200/weyl_600K_diag.py ../weyl_vs_T_h200/weyl_trajectories.dat
For each node: multiplet spread at its 600 K position (full fit and both halves),
the gap minimum re-converged from its 300 K position, charges on spheres of several
radii, and the nearest other node (including symmetry copies) at 600 K.
"""
import os, sys, numpy as np, phonopy, spglib
from collections import defaultdict
from ase import Atoms
from vasp_grace.condmat import phonon_tools as pt
from vasp_grace.condmat.topological_phonons import (node_chern, follow_node, symmetry_images,
                                                    _multiplet_spread)

rows = defaultdict(dict)
for line in open(sys.argv[1]):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split()
    lo, hi = map(int, p[2].split("-"))
    rows[int(p[0])][int(p[1])] = dict(q=np.array(p[3:6], float), bands=(lo, hi), charges=p[9])

ph = {t: pt.PhonopyPhonons(phonopy.load(f"fc_600K_{t}.yaml.xz")) for t in ("full", "a", "b")}
P = ph["full"].phonon.primitive
prim = Atoms(numbers=P.numbers, cell=P.cell, scaled_positions=P.scaled_positions, pbc=True)
rec = np.asarray(prim.cell.reciprocal())
ds = spglib.get_symmetry_dataset((prim.cell.array, prim.get_scaled_positions(), prim.numbers),
                                 symprec=1e-3)
dist = lambda a, b: np.linalg.norm(((a - b + 0.5) % 1.0 - 0.5) @ rec)

# every 600 K node with its symmetry copies, to find neighbours
all600 = []
for i, r in rows[600].items():
    node = {"q": r["q"], "bands": r["bands"], "charges": {}, "band_chern": []}
    all600 += [(i, r["q"])] + [(i, img["q"]) for img in symmetry_images(node, ds.rotations)]

ids = [int(a) for a in sys.argv[2:]] or list(range(45, 78))


def check(i):
    r6, r3 = rows[600][i], rows[300][i]
    lo, hi = r6["bands"]
    q = r6["q"]
    spreads = [_multiplet_spread(ph[t], q, lo, hi) for t in ("full", "a", "b")]
    qr, gap = follow_node(ph["full"], r3["q"], lo, hi)
    ch = [str(node_chern(ph["full"], prim, q, lo, hi, radius=rad)["charges"].get(lo, "?"))
          for rad in (0.0025, 0.005, 0.01, 0.02)]
    # nearest other point (symmetry copies at the same q are the same point), with its 300 K charge
    j, dj = min(((j, dist(q, qq)) for j, qq in all600 if dist(q, qq) > 1e-4), key=lambda x: x[1])
    return (f"{i:3d} ({q[0]:.4f},{q[1]:.4f},{q[2]:.4f})  "
            f"{spreads[0]:.4f}/{spreads[1]:.4f}/{spreads[2]:.4f}  {gap:10.4f} {dist(qr, q):9.4f}  "
            f"{'/'.join(ch):>30}   ({j}, {dj:.4f}, C300 {rows[300][j]['charges']})")


print(f"{'id':>3} {'q600':>26} {'spread full/a/b (THz)':>24} {'reconv gap':>10} "
      f"{'|dq| 1/Å':>9}  charges r=0.0025/0.005/0.01/0.02   nearest other point (id, 1/Å, charge 300 K)")
from multiprocessing import Pool
with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 1))) as pool:
    for line in pool.imap(check, ids):
        print(line, flush=True)
