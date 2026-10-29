import time, numpy as np
from ase.io import read
from vasp_grace.calculator import load_calculator
from vasp_grace.condmat.phonon_tools import phonopy_phonons, eigensystem

calc = load_calculator("GRACE-2L-OAM", verbose=False)
prim = read("POSCAR")
for n in (3, 4, 6):
    a = prim.repeat(n); a.calc = calc; rng = np.random.default_rng(0)
    for _ in range(3):
        a.positions += 0.01 * rng.standard_normal(a.positions.shape); a.get_forces()
    t = time.perf_counter()
    for _ in range(10):
        a.positions += 0.01 * rng.standard_normal(a.positions.shape); a.get_forces()
    dt = (time.perf_counter() - t) / 10
    print(f"{len(a):5d} atoms: {dt*1e3:7.1f} ms per force call")
ph = phonopy_phonons(prim, calc, (3, 3, 3), 0.015, save=None)
print("Γ", np.round(eigensystem(ph, [0, 0, 0])[0][[6, 9, 12, 17, 21]], 3))
print("R", np.round(eigensystem(ph, [0.5, 0.5, 0.5])[0][[0, 4, 8, 12, 16, 20]], 3))
