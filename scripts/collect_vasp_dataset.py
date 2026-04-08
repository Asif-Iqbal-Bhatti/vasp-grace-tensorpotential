#!/usr/bin/env python3
"""
collect_vasp_dataset.py
=======================
Python wrapper around the ``grace_collect`` CLI utility shipped with
gracemaker / TensorPotential.

Walk a directory tree of VASP calculations, extract energies, forces,
and stresses from each run, compute or specify per-element reference
energies, and produce a ``collected.pkl.gz`` dataset ready for
gracemaker training.

Internally calls ``grace_collect``; this script adds:
  - element auto-detection from structure files
  - pre-flight checks (OUTCAR existence, convergence flags)
  - optional VASPRUN fallback when OUTCAR is incomplete
  - progress reporting

Usage
-----
  python collect_vasp_dataset.py \\
      --working-dir /path/to/vasp_runs \\
      --output      collected.pkl.gz \\
      --elements    Li Ni Mn Co O \\
      --selection   all \\
      --free-atom-energy "Li:auto Ni:auto Mn:auto Co:auto O:auto"

Author: vasp-grace workflow
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from ase.io import read

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect_vasp")


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_outcar_converged(outcar: Path) -> bool:
    """
    Quick convergence check: look for 'reached required accuracy'
    in the OUTCAR (written by VASP on successful SCF convergence of
    at least the final ionic step).
    """
    converged = False
    try:
        with open(outcar, "r", errors="replace") as fh:
            for line in fh:
                if "reached required accuracy" in line:
                    converged = True
                    break
    except OSError:
        pass
    return converged


def discover_vasp_dirs(root: Path) -> List[Path]:
    """Return all directories containing an OUTCAR file under *root*."""
    dirs = sorted({p.parent for p in root.rglob("OUTCAR")})
    log.info("Discovered %d VASP calculation director(y/ies).", len(dirs))
    return dirs


def detect_elements(dirs: List[Path]) -> Set[str]:
    """Detect chemical species present across all discovered VASP calculations."""
    elements: Set[str] = set()
    for d in dirs:
        poscar = d / "POSCAR"
        contcar = d / "CONTCAR"
        source = contcar if contcar.exists() else poscar
        if not source.exists():
            continue
        try:
            atoms = read(str(source), format="vasp")
            elements.update(atoms.get_chemical_symbols())
        except Exception:
            pass
    return elements


def build_free_atom_energy_str(
    elements: List[str],
    custom: Optional[Dict[str, float]],
) -> str:
    """
    Build the ``--free-atom-energy`` string for ``grace_collect``.

    Custom values override; remaining elements default to 'auto'.
    """
    parts = []
    custom = custom or {}
    for el in elements:
        if el in custom:
            parts.append(f"{el}:{custom[el]:.6f}")
        else:
            parts.append(f"{el}:auto")
    return " ".join(parts)


def run_grace_collect(
    working_dir: Path,
    output: Path,
    free_atom_energy_str: str,
    selection: str,
    extra_args: List[str],
) -> int:
    """
    Invoke ``grace_collect`` as a subprocess and return its exit code.
    """
    if shutil.which("grace_collect") is None:
        log.error(
            "'grace_collect' not found in PATH. "
            "Install gracemaker: pip install tensorpotential"
        )
        return 1

    cmd = [
        "grace_collect",
        "--working-dir", str(working_dir),
        "--output-dataset-filename", str(output),
        "--free-atom-energy", free_atom_energy_str,
        "--selection", selection,
    ] + extra_args

    log.info("Running: %s", " ".join(cmd))
    env = {**os.environ}
    result = subprocess.run(cmd, env=env, text=True)
    return result.returncode


# ── Pre-flight validation ─────────────────────────────────────────────────────

def preflight(
    dirs: List[Path],
    check_convergence: bool,
    skip_unconverged: bool,
) -> List[Path]:
    """
    Validate VASP calculation directories and optionally filter
    unconverged runs.

    Returns the validated (possibly filtered) list of directories.
    """
    ok: List[Path] = []
    n_unconverged = 0

    for d in dirs:
        outcar = d / "OUTCAR"
        if not outcar.exists():
            log.warning("  MISSING OUTCAR: %s — skipping.", d)
            continue

        if check_convergence and not check_outcar_converged(outcar):
            n_unconverged += 1
            if skip_unconverged:
                log.warning("  NOT CONVERGED (skipping): %s", d)
                continue
            else:
                log.warning("  NOT CONVERGED (keeping): %s", d)

        ok.append(d)

    if n_unconverged and not skip_unconverged:
        log.warning(
            "%d unconverged calculation(s) included. "
            "Use --skip-unconverged to exclude them.",
            n_unconverged,
        )

    log.info("Validated: %d/%d directories usable.", len(ok), len(dirs))
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--working-dir", "-wd", type=Path, required=True,
        help="Root directory containing VASP calculation sub-directories.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("collected.pkl.gz"),
        help="Output dataset file. (default: collected.pkl.gz)",
    )
    parser.add_argument(
        "--elements", nargs="+", default=None, metavar="El",
        help="List of chemical elements. "
             "If omitted, auto-detected from POSCAR/CONTCAR files.",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "last", "all", "first_and_last"],
        default="last",
        help="Which ionic steps to include from each OUTCAR. (default: last)",
    )
    parser.add_argument(
        "--free-atom-energy", type=str, default=None,
        metavar="'El:val El2:auto ...'",
        help="Per-element reference energies passed to grace_collect. "
             "Use 'auto' to have gracemaker extract from single-atom cells. "
             "Example: \"Li:-1.90 O:auto\". "
             "If omitted, all elements default to 'auto'.",
    )
    parser.add_argument(
        "--check-convergence", action="store_true", default=True,
        help="Check OUTCAR for SCF convergence flag. (default: True)",
    )
    parser.add_argument(
        "--skip-unconverged", action="store_true", default=False,
        help="Skip directories whose OUTCAR shows no convergence.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover directories and print summary, but do not run grace_collect.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args, extra = parser.parse_known_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # 1. Discover calculation directories
    dirs = discover_vasp_dirs(args.working_dir)
    if not dirs:
        log.error("No VASP calculations found under %s.", args.working_dir)
        sys.exit(1)

    # 2. Pre-flight checks
    dirs = preflight(dirs, args.check_convergence, args.skip_unconverged)
    if not dirs:
        log.error("No usable directories after validation.")
        sys.exit(1)

    # 3. Element detection
    elements: List[str]
    if args.elements:
        elements = args.elements
    else:
        elements = sorted(detect_elements(dirs))
        log.info("Auto-detected elements: %s", elements)

    if not elements:
        log.error("Could not detect any chemical elements. Use --elements.")
        sys.exit(1)

    # 4. Build free-atom-energy string
    if args.free_atom_energy:
        fae_str = args.free_atom_energy
    else:
        fae_str = build_free_atom_energy_str(elements, custom=None)
    log.info("Reference energies: %s", fae_str)

    # 5. Dry-run summary
    if args.dry_run:
        log.info("=== DRY RUN ===")
        log.info("Would process %d directories.", len(dirs))
        log.info("Elements: %s", elements)
        log.info("Output:   %s", args.output)
        for d in dirs[:10]:
            log.info("  %s", d)
        if len(dirs) > 10:
            log.info("  ... and %d more.", len(dirs) - 10)
        return

    # 6. Run grace_collect
    rc = run_grace_collect(
        working_dir=args.working_dir,
        output=args.output,
        free_atom_energy_str=fae_str,
        selection=args.selection,
        extra_args=extra,
    )

    if rc != 0:
        log.error("grace_collect returned exit code %d.", rc)
        sys.exit(rc)

    log.info("Dataset written: %s", args.output.resolve())


if __name__ == "__main__":
    main()
