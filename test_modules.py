#!/usr/bin/env python3
"""
test_modules.py

Lightweight integration test suite for vasp-grace-tensorpotential modules.

Strategy
--------
- Modules that need no GRACE model (topology, dislocation, moire, lihopping)
  are run end-to-end with a synthetic FCC-Al POSCAR.
- GRACE-dependent modules (active_learning, montecarlo, thermal_conductivity,
  topological_phonons, phonon_berry) are tested for correct imports and
  argument parsing (--help exits 0 by default).
- A summary table is printed at the end.

Run:
    python test_modules.py
"""

import os
import sys
import subprocess
import tempfile
import shutil
import textwrap
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer WSL miniconda Python (has GRACE) when running on Windows
WSL_PYTHON = "/home/asifem/miniconda3/bin/python"
_wsl_check = subprocess.run(
    ["wsl", WSL_PYTHON, "--version"],
    capture_output=True, text=True
) if sys.platform == "win32" else None
USE_WSL = (sys.platform == "win32" and
           _wsl_check is not None and
           _wsl_check.returncode == 0)

PYTHON     = sys.executable
GRACE_MODEL = "GRACE-2L-OAM"

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_wsl_path(p):
    """Convert a Windows path to /mnt/... for WSL."""
    if p and len(p) > 1 and p[1] == ":":
        drive = p[0].lower()
        rest  = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p


def run(cmd, cwd=None, timeout=120, force_wsl=False):
    """Run a subprocess, return (returncode, stdout+stderr).
    When USE_WSL is True or force_wsl=True, routes through WSL."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"

    if USE_WSL or force_wsl:
        # Translate all Windows paths in cmd to /mnt/... form
        wsl_cmd = [_to_wsl_path(str(c)) for c in cmd]
        # Replace sys.executable with WSL Python
        if wsl_cmd[0] == _to_wsl_path(PYTHON) or wsl_cmd[0] == PYTHON:
            wsl_cmd[0] = WSL_PYTHON
        wsl_cwd = _to_wsl_path(cwd) if cwd else None
        # Build env string for WSL
        env_pairs = " ".join(f'{k}="{v}"' for k, v in [
            ("PYTHONIOENCODING", "utf-8"),
            ("PYTHONUTF8", "1"),
            ("TF_CPP_MIN_LOG_LEVEL", "3"),
            ("TF_ENABLE_ONEDNN_OPTS", "0"),
        ])
        inner = " ".join(f'"{c}"' for c in wsl_cmd)
        bash_cmd = f"cd {wsl_cwd or '~'} && {env_pairs} {inner}"
        proc = subprocess.run(
            ["wsl", "bash", "-c", bash_cmd],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout
        )
    else:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env, timeout=timeout
        )
    return proc.returncode, proc.stdout + proc.stderr


def record(name, status, detail=""):
    tag = PASS if status == "pass" else (FAIL if status == "fail" else SKIP)
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    results.append((name, status, detail))


def section(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic structures
# ──────────────────────────────────────────────────────────────────────────────

FCC_AL_POSCAR = textwrap.dedent("""\
    Al FCC
    1.0
       4.050000  0.000000  0.000000
       0.000000  4.050000  0.000000
       0.000000  0.000000  4.050000
    Al
    4
    Direct
     0.000000  0.000000  0.000000
     0.500000  0.500000  0.000000
     0.500000  0.000000  0.500000
     0.000000  0.500000  0.500000
""")

# Simple BCC Li for hop tests
BCC_LI_POSCAR = textwrap.dedent("""\
    Li BCC
    1.0
       3.510000  0.000000  0.000000
       0.000000  3.510000  0.000000
       0.000000  0.000000  3.510000
    Li
    2
    Direct
     0.000000  0.000000  0.000000
     0.500000  0.500000  0.500000
""")

# Minimal 2D hexagonal monolayer (graphene-like, 1 Å thick slab)
HEX_MONO_POSCAR = textwrap.dedent("""\
    Hex monolayer
    1.0
       2.460000  0.000000  0.000000
      -1.230000  2.130000  0.000000
       0.000000  0.000000 20.000000
    C
    2
    Direct
     0.000000  0.000000  0.500000
     0.333333  0.666667  0.500000
""")


def make_xdatcar(poscar_content, n_frames=20, rng_seed=42):
    """Generate a minimal XDATCAR with small random displacements."""
    from ase.io import read
    from ase.io import write
    import io

    atoms = read(io.StringIO(poscar_content), format="vasp")
    rng   = np.random.default_rng(rng_seed)

    lines = ["XDATCAR generated by test_modules.py\n",
             "1.0\n"]
    cell = atoms.get_cell()
    for row in cell:
        lines.append(f"  {row[0]:12.6f}  {row[1]:12.6f}  {row[2]:12.6f}\n")
    syms = atoms.get_chemical_symbols()
    unique = list(dict.fromkeys(syms))
    counts = [syms.count(s) for s in unique]
    lines.append("  ".join(unique) + "\n")
    lines.append("  ".join(str(c) for c in counts) + "\n")

    base_frac = atoms.get_scaled_positions()
    for f in range(n_frames):
        lines.append(f"Direct configuration=  {f+1}\n")
        disp = rng.normal(0, 0.02, base_frac.shape)
        for pos in (base_frac + disp) % 1.0:
            lines.append(f"  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}\n")

    return "".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dependency check
# ──────────────────────────────────────────────────────────────────────────────

section("1. Dependency check")

if USE_WSL:
    print(f"  (routing through WSL: {WSL_PYTHON})")

for pkg, import_str in [
    ("ase",         "import ase"),
    ("numpy",       "import numpy"),
    ("scipy",       "import scipy"),
    ("matplotlib",  "import matplotlib"),
    ("tensorpotential", "import tensorpotential"),
    ("gudhi",       "import gudhi"),
]:
    rc, out = run([PYTHON, "-c", import_str])
    if rc == 0:
        record(pkg, "pass", "installed")
    else:
        record(pkg, "skip", "not installed (some tests will be skipped)")

HAS_TP    = any(r[0] == "tensorpotential" and r[1] == "pass" for r in results)
HAS_GUDHI = any(r[0] == "gudhi"           and r[1] == "pass" for r in results)


# ──────────────────────────────────────────────────────────────────────────────
# 2. topology.py
# ──────────────────────────────────────────────────────────────────────────────

section("2. topology.py  (persistent homology)")

with tempfile.TemporaryDirectory() as tmp:
    poscar = os.path.join(tmp, "POSCAR")
    with open(poscar, "w") as f:
        f.write(FCC_AL_POSCAR)

    # Single structure
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "topology.py"),
                   "--poscar", poscar, "--rmax", "6.0", "--npoints", "30"],
                  cwd=tmp)
    if rc == 0 and os.path.exists(os.path.join(tmp, "betti_curves.dat")):
        record("single structure analysis", "pass")
    else:
        record("single structure analysis", "fail", out[-300:])

    # Compare mode (use same file for both)
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "topology.py"),
                   "--poscar", poscar, "--compare", poscar,
                   "--rmax", "6.0", "--npoints", "30"],
                  cwd=tmp)
    if rc == 0 and os.path.exists(os.path.join(tmp, "betti_curves_compare.dat")):
        record("compare mode (bulk vs GB)", "pass")
    else:
        record("compare mode (bulk vs GB)", "fail", out[-300:])

    # PBC mode
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "topology.py"),
                   "--poscar", poscar, "--rmax", "6.0", "--npoints", "20", "--pbc"],
                  cwd=tmp)
    record("PBC distance matrix", "pass" if rc == 0 else "fail", out[-200:] if rc else "")

    # XDATCAR trajectory mode
    xdat = make_xdatcar(FCC_AL_POSCAR, n_frames=5)
    xdatcar = os.path.join(tmp, "XDATCAR")
    with open(xdatcar, "w") as f:
        f.write(xdat)
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "topology.py"),
                   "--xdatcar", xdatcar, "--rmax", "6.0", "--npoints", "20"],
                  cwd=tmp)
    record("XDATCAR trajectory mode", "pass" if rc == 0 else "fail", out[-200:] if rc else "")


# ──────────────────────────────────────────────────────────────────────────────
# 3. dislocation.py
# ──────────────────────────────────────────────────────────────────────────────

section("3. dislocation.py  (Volterra / Stroh)")

with tempfile.TemporaryDirectory() as tmp:
    poscar = os.path.join(tmp, "POSCAR")
    with open(poscar, "w") as f:
        f.write(FCC_AL_POSCAR)

    for dtype in ["screw", "edge"]:
        rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "dislocation.py"),
                       "--poscar", poscar, "--type", dtype, "--burgers", "2.86"],
                      cwd=tmp)
        ok = rc == 0 and os.path.exists(os.path.join(tmp, "POSCAR_dislocation"))
        record(f"isotropic Volterra — {dtype}", "pass" if ok else "fail",
               out[-300:] if not ok else "")

    # Edge with Poisson ratio
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "dislocation.py"),
                   "--poscar", poscar, "--type", "edge",
                   "--burgers", "2.86", "--poisson", "0.33"],
                  cwd=tmp)
    record("edge with Poisson ratio", "pass" if rc == 0 else "fail", out[-200:] if rc else "")

    # Miller index input
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "dislocation.py"),
                   "--poscar", poscar, "--type", "screw", "--burgers", "2.86",
                   "--hkl", "1", "1", "0", "--uvw", "1", "-1", "0"],
                  cwd=tmp)
    record("Miller index (hkl/uvw) input", "pass" if rc == 0 else "fail", out[-200:] if rc else "")

    # Synthetic elastic constants file for Stroh test
    elastic_dat = os.path.join(tmp, "ELASTIC_Cij_GPa.dat")
    with open(elastic_dat, "w") as f:
        # Al elastic constants (GPa)
        C = [[108, 62, 62, 0, 0, 0],
             [62, 108, 62, 0, 0, 0],
             [62,  62,108, 0, 0, 0],
             [ 0,   0,  0,28, 0, 0],
             [ 0,   0,  0, 0,28, 0],
             [ 0,   0,  0, 0, 0,28]]
        for row in C:
            f.write("  ".join(f"{v:8.2f}" for v in row) + "\n")
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "dislocation.py"),
                   "--poscar", poscar, "--type", "edge", "--burgers", "2.86",
                   "--method", "anisotropic", "--elastic", elastic_dat],
                  cwd=tmp)
    record("anisotropic Stroh formalism", "pass" if rc == 0 else "fail", out[-300:] if rc else "")


# ──────────────────────────────────────────────────────────────────────────────
# 4. moire.py
# ──────────────────────────────────────────────────────────────────────────────

section("4. moire.py  (CSL twisted bilayer)")

with tempfile.TemporaryDirectory() as tmp:
    poscar = os.path.join(tmp, "POSCAR")
    with open(poscar, "w") as f:
        f.write(HEX_MONO_POSCAR)

    # Scan mode (no POSCAR needed for output, but CLI requires --poscar)
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "moire.py"),
                   "--poscar", poscar, "--scan", "--m_max", "6"],
                  cwd=tmp)
    record("CSL angle scan (hex)", "pass" if rc == 0 else "fail", out[-200:] if rc else "")

    # Small (m,n) = (2,3): 27 cells per layer — manageable
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "moire.py"),
                   "--poscar", poscar, "--m", "2", "--n", "3",
                   "--gap", "3.35", "--vacuum", "15.0"],
                  cwd=tmp)
    ok = rc == 0 and os.path.exists(os.path.join(tmp, "POSCAR_moire"))
    record("bilayer build (m=2, n=3)", "pass" if ok else "fail", out[-300:] if not ok else "")

    # Square lattice scan
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "moire.py"),
                   "--poscar", poscar, "--scan", "--m_max", "6", "--lattice", "square"],
                  cwd=tmp)
    record("CSL angle scan (square)", "pass" if rc == 0 else "fail", out[-200:] if rc else "")


# ──────────────────────────────────────────────────────────────────────────────
# 5. lihopping.py
# ──────────────────────────────────────────────────────────────────────────────

section("5. lihopping.py  (ion hop detection)")

with tempfile.TemporaryDirectory() as tmp:
    poscar = os.path.join(tmp, "POSCAR")
    with open(poscar, "w") as f:
        f.write(BCC_LI_POSCAR)

    xdatcar = os.path.join(tmp, "XDATCAR")
    with open(xdatcar, "w") as f:
        f.write(make_xdatcar(BCC_LI_POSCAR, n_frames=40, rng_seed=7))

    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "lihopping.py"),
                   "--traj", xdatcar, "--ref", poscar,
                   "--species", "Li", "--timestep", "2.0", "--min_sojourn", "2"],
                  cwd=tmp)
    ok = rc == 0 and os.path.exists(os.path.join(tmp, "hop_events.dat"))
    record("hop detection (single trajectory)", "pass" if ok else "fail", out[-300:] if not ok else "")

    # Arrhenius with two identical trajectories (same rate, just checking no crash)
    xdat2 = os.path.join(tmp, "XDATCAR2")
    shutil.copy(xdatcar, xdat2)
    rc, out = run([PYTHON, os.path.join(SCRIPT_DIR, "lihopping.py"),
                   "--arrhenius",
                   "--temps", "600", "800",
                   "--trajs", xdatcar, xdat2,
                   "--ref", poscar, "--species", "Li", "--timestep", "2.0"],
                  cwd=tmp)
    record("Arrhenius analysis (2 temperatures)", "pass" if rc == 0 else "fail",
           out[-300:] if rc else "")


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI / import tests for GRACE-dependent modules
# ──────────────────────────────────────────────────────────────────────────────

section("6. GRACE-dependent modules  (CLI + import validation)")

grace_modules = [
    ("active_learning.py",      ["--help"]),
    ("montecarlo.py",           ["--help"]),
    ("thermal_conductivity.py", ["--help"]),
    ("topological_phonons.py",  ["--help"]),
    ("phonon_berry.py",         ["--help"]),
]

for script, args in grace_modules:
    path = os.path.join(SCRIPT_DIR, script)
    rc, out = run([PYTHON, path] + args)
    # argparse --help exits with code 0
    record(f"{script} --help", "pass" if rc == 0 else "fail", out[-100:] if rc else "")

# Syntax check — translate Windows paths to WSL paths when needed
for script in ["active_learning.py", "montecarlo.py", "thermal_conductivity.py",
               "topological_phonons.py", "phonon_berry.py"]:
    path = os.path.join(SCRIPT_DIR, script)
    fpath = _to_wsl_path(path) if USE_WSL else path
    check = (
        "import ast\n"
        f"src = open('{fpath}', encoding='utf-8').read()\n"
        "ast.parse(src)\n"
        "print('syntax OK')\n"
    )
    rc, out = run([PYTHON, "-c", check])
    record(f"{script} syntax check", "pass" if rc == 0 else "fail",
           out.strip() if rc == 0 else out[-200:])


# ──────────────────────────────────────────────────────────────────────────────
# 7. main.py import / syntax
# ──────────────────────────────────────────────────────────────────────────────

section("7. main.py  (syntax + structure)")

for script in ["main.py", "__initi__.py"]:
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(path):
        record(script, "skip", "file not found")
        continue
    fpath = _to_wsl_path(path) if USE_WSL else path
    check = (
        "import ast\n"
        f"src = open('{fpath}', encoding='utf-8').read()\n"
        "ast.parse(src)\n"
        "print('syntax OK')\n"
    )
    rc, out = run([PYTHON, "-c", check])
    record(f"{script} syntax", "pass" if rc == 0 else "fail",
           out.strip() if rc == 0 else out[-200:])


# ──────────────────────────────────────────────────────────────────────────────
# 8. GRACE end-to-end tests (require tensorpotential)
# ──────────────────────────────────────────────────────────────────────────────

section("8. GRACE end-to-end tests  (active_learning, montecarlo, thermal_conductivity)")

if not HAS_TP:
    record("all GRACE end-to-end tests", "skip", "tensorpotential not available")
else:
    # Use the WSL project path directly so scripts find each other
    WSL_PROJ = _to_wsl_path(SCRIPT_DIR) if USE_WSL else SCRIPT_DIR

    with tempfile.TemporaryDirectory() as tmp:
        tmp_wsl = _to_wsl_path(tmp) if USE_WSL else tmp

        poscar = os.path.join(tmp, "POSCAR")
        with open(poscar, "w") as f:
            f.write(FCC_AL_POSCAR)
        poscar_arg = _to_wsl_path(poscar) if USE_WSL else poscar

        # ── active_learning: screen 1 structure with single model (used twice as committee)
        script = os.path.join(SCRIPT_DIR, "active_learning.py")
        script_arg = _to_wsl_path(script) if USE_WSL else script
        rc, out = run(
            [PYTHON, script_arg,
             "--poscar", poscar_arg,
             "--models", GRACE_MODEL, GRACE_MODEL,
             "--threshold", "999.0"],   # high threshold so nothing gets flagged
            cwd=tmp, timeout=180
        )
        ok = rc == 0 and os.path.exists(os.path.join(tmp, "uncertainty_log.dat"))
        record("active_learning — single POSCAR, committee of 2", "pass" if ok else "fail",
               out[-400:] if not ok else "")

        # ── montecarlo: 100 steps (fast)
        script = os.path.join(SCRIPT_DIR, "montecarlo.py")
        script_arg = _to_wsl_path(script) if USE_WSL else script
        rc, out = run(
            [PYTHON, script_arg,
             "--poscar", poscar_arg,
             "--model", GRACE_MODEL,
             "--temperature", "300",
             "--steps", "100",
             "--save_interval", "20"],
            cwd=tmp, timeout=240
        )
        ok = rc == 0 and os.path.exists(os.path.join(tmp, "MC_energies.dat"))
        record("montecarlo — 100 steps at 300 K", "pass" if ok else "fail",
               out[-400:] if not ok else "")

        # ── thermal_conductivity: 500 steps (minimal but exercises the full pipeline)
        script = os.path.join(SCRIPT_DIR, "thermal_conductivity.py")
        script_arg = _to_wsl_path(script) if USE_WSL else script
        rc, out = run(
            [PYTHON, script_arg,
             "--poscar", poscar_arg,
             "--model", GRACE_MODEL,
             "--temperature", "300",
             "--steps", "500",
             "--equil_frac", "0.4",
             "--nslabs", "4"],
            cwd=tmp, timeout=300
        )
        ok = rc == 0 and os.path.exists(os.path.join(tmp, "kappa_summary.txt"))
        record("thermal_conductivity — 500 MD steps", "pass" if ok else "fail",
               out[-400:] if not ok else "")


section("9. Phonon modules  (topological_phonons, phonon_berry)")

if not HAS_TP:
    record("phonon end-to-end tests", "skip", "tensorpotential not available")
else:
    with tempfile.TemporaryDirectory() as tmp:
        poscar = os.path.join(tmp, "POSCAR")
        with open(poscar, "w") as f:
            f.write(FCC_AL_POSCAR)
        poscar_arg = _to_wsl_path(poscar) if USE_WSL else poscar

        # Run phonon finite displacements (2x2x2 supercell, minimal k-path)
        # This is shared by both topological_phonons and phonon_berry
        print("  Building phonon force constants (2x2x2 supercell)...")
        setup_code = f"""
import sys, os
sys.path.insert(0, '{_to_wsl_path(SCRIPT_DIR) if USE_WSL else SCRIPT_DIR}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from ase.io import read
from ase.phonons import Phonons
ph_script = __import__('topological_phonons')
atoms = read('{poscar_arg}', format='vasp')
calc  = ph_script.load_model('{GRACE_MODEL}')
ph    = Phonons(atoms, calc, supercell=(2,2,2), delta=0.015, name='phonon')
ph.run()
ph.read(method='Frederiksen', symmetrize=3, acoustic=True, cutoff=None, born=False)
print('phonon force constants built')
"""
        script_run = _to_wsl_path(os.path.join(SCRIPT_DIR, "topological_phonons.py")) if USE_WSL \
                     else os.path.join(SCRIPT_DIR, "topological_phonons.py")
        rc, out = run([PYTHON, "-c", setup_code], cwd=tmp, timeout=600)
        if rc != 0:
            record("phonon force constants (2x2x2)", "fail", out[-400:])
        else:
            record("phonon force constants (2x2x2)", "pass")

            # topological_phonons: use cached data
            rc, out = run(
                [PYTHON, script_run,
                 "--poscar", poscar_arg,
                 "--model", GRACE_MODEL,
                 "--supercell", "2", "2", "2",
                 "--path", "GX",
                 "--nkpts", "20",
                 "--tol", "2.0"],
                cwd=tmp, timeout=300
            )
            ok = rc == 0 and os.path.exists(os.path.join(tmp, "phonon_crossings.dat"))
            record("topological_phonons — GX path, 20 k-pts", "pass" if ok else "fail",
                   out[-400:] if not ok else "")

            # phonon_berry: Zak phases
            script_berry = _to_wsl_path(os.path.join(SCRIPT_DIR, "phonon_berry.py")) if USE_WSL \
                           else os.path.join(SCRIPT_DIR, "phonon_berry.py")
            rc, out = run(
                [PYTHON, script_berry,
                 "--poscar", poscar_arg,
                 "--model", GRACE_MODEL,
                 "--supercell", "2", "2", "2",
                 "--path", "GXG",
                 "--nkpts", "20"],
                cwd=tmp, timeout=300
            )
            ok = rc == 0 and os.path.exists(os.path.join(tmp, "phonon_zak.dat"))
            record("phonon_berry — Zak phases GXG, 20 k-pts", "pass" if ok else "fail",
                   out[-400:] if not ok else "")


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  TEST SUMMARY")
print(f"{'='*60}")

passed = sum(1 for _, s, _ in results if s == "pass")
failed = sum(1 for _, s, _ in results if s == "fail")
skipped = sum(1 for _, s, _ in results if s == "skip")
total  = len(results)

print(f"  Total : {total}   Passed : {passed}   Failed : {failed}   Skipped : {skipped}")
print()

if failed:
    print("  Failed tests:")
    for name, status, detail in results:
        if status == "fail":
            print(f"    x  {name}")
            if detail:
                for line in detail.strip().splitlines()[-5:]:
                    print(f"       {line}")

print(f"\n{'='*60}\n")
sys.exit(0 if failed == 0 else 1)
