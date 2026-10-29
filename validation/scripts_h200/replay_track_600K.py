"""
End-to-end check of track_nodes at 600 K without MD (CPU only): effective_phonons is
stubbed to return the saved 600 K force constants (fc_dump.py), and the nodes start
from their recorded 300 K positions with their 0 K charges. Run from
the directory holding the fc_600K_* files (validation/weyl_600K_diagnosis for seed 0):
    python ../scripts_h200/replay_track_600K.py ../weyl_vs_T_h200/weyl_trajectories.dat
"""
import sys, json, types, numpy as np, phonopy
from ase import Atoms
from vasp_grace.condmat import topological_phonons as tp
from vasp_grace.condmat.phonon_tools import PhonopyPhonons

ph = {t: PhonopyPhonons(phonopy.load(f"fc_600K_{t}.yaml.xz")) for t in ("full", "a", "b")}
P = ph["full"].phonon.primitive
prim = Atoms(numbers=P.numbers, cell=P.cell, scaled_positions=P.scaled_positions, pbc=True)
info = json.load(open("fc_600K_info.json"))
tp.effective_phonons = lambda *a, **k: {"ph": ph["full"], "halves": (ph["a"], ph["b"]),
                                        "prim": prim, "info": info}
rows = {}
for line in open(sys.argv[1]):
    if line.startswith("#") or not line.strip():
        continue
    p = line.split()
    lo, hi = map(int, p[2].split("-"))
    ch = {int(g): int(c) for g, c in (x.split(":") for x in p[9].split(","))} if p[9] != "-" else {}
    rows[(int(p[0]), int(p[1]))] = dict(q=np.array(p[3:6], float), bands=(lo, hi), freq=float(p[6]), charges=ch)
ids = sorted(i for T, i in rows if T == 0)
base = [{"q": rows[(300, i)]["q"], "bands": rows[(0, i)]["bands"], "charges": rows[(0, i)]["charges"],
         "frequency_THz": rows[(300, i)]["freq"], "multiplicity": 2} for i in ids]
args = types.SimpleNamespace(supercell=[3, 3, 3], md_timestep=2, npt_steps=0, md_equil=0,
                             md_snapshots=0, md_every=0, no_npt=False, seed=0, degen_tol=0.05)
out, _, events, _ = tp.track_nodes(base, [600], prim, None, args,
                                   dict(radius=0.01, n_theta=16, tol=0.05), ph["full"])
from collections import Counter
print(Counter(r["status"].split(" ")[0].rstrip(":,") for r in out if r["T"] == 600))
print("\n".join(events))
