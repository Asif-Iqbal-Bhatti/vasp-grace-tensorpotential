"""
Run effective_phonons at T with the grace-topo-phonons defaults (150 snapshots) and
save the effective force constants of the full fit and of both half-fits as phonopy
files, for CPU-only node analysis (weyl_600K_diag.py). Run from the directory holding POSCAR:
    python fc_dump.py 600 [seed]   -> fc_600K_full.yaml, fc_600K_a.yaml, fc_600K_b.yaml, fc_600K_info.json
Compress them with `xz -9` (16 MB -> 1.7 MB each; phonopy.load reads .yaml.xz directly).
"""
import sys, json
from ase.io import read
from vasp_grace.calculator import load_calculator
from vasp_grace.condmat import phonon_tools as pt

T = float(sys.argv[1]); seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
prim = read("POSCAR"); calc = load_calculator("GRACE-2L-OAM")
eff = pt.effective_phonons(prim, calc, T, (3, 3, 3), n_snapshots=150, sample_every=100, npt=True, seed=seed)
for tag, ph in zip(("full", "a", "b"), (eff["ph"], *eff["halves"])):
    ph.phonon.save(f"fc_{T:g}K_{tag}.yaml", settings={"force_constants": True})
json.dump(eff["info"], open(f"fc_{T:g}K_info.json", "w"), indent=1, default=str)
print("info", eff["info"], flush=True)
