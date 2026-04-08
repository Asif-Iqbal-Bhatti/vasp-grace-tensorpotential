#!/usr/bin/env python3
"""
vasp_to_extxyz.py
=================
Convert VASP OUTCAR or vasprun.xml files to extended XYZ format
suitable for gracemaker / GRACE MLIP training.

Handles:
  - Single OUTCAR  (static, MD, or relaxation trajectory)
  - Single vasprun.xml
  - Directory tree  (walks recursively, finds all OUTCARs / vasprun.xml)
  - Proper unit conversion: VASP stress (kBar) → eV/Å³ virial
  - Configurable energy / forces / stress key names
  - Structure selection: first, last, all, first_and_last

Usage
-----
  python vasp_to_extxyz.py --input-dir /path/to/vasp_runs \\
                            --output dataset.extxyz \\
                            --selection all \\
                            --verbose

  python vasp_to_extxyz.py --input OUTCAR --output out.extxyz

Author: vasp-grace workflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.units import GPa

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vasp_to_extxyz")


# ── Constants ─────────────────────────────────────────────────────────────────
# VASP reports stress in kBar; 1 kBar = 0.1 GPa = 0.1/GPa eV/Å³ in ASE units
KBAR_TO_EV_ANG3 = 1e-1 / GPa  # ≈ 6.2415e-4  eV/Å³ per kBar


# ── Core helpers ─────────────────────────────────────────────────────────────

def _apply_selection(frames: List[Atoms], selection: str) -> List[Atoms]:
    """Return a subset of frames according to *selection* strategy."""
    if not frames:
        return frames
    if selection == "all":
        return frames
    elif selection == "first":
        return [frames[0]]
    elif selection == "last":
        return [frames[-1]]
    elif selection == "first_and_last":
        return [frames[0], frames[-1]] if len(frames) > 1 else [frames[0]]
    else:
        raise ValueError(f"Unknown selection strategy: {selection!r}")


def _tag_frame(
    atoms: Atoms,
    source: str,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    config_type: str,
    convert_stress: bool,
) -> Optional[Atoms]:
    """
    Attach energy / forces / stress labels expected by gracemaker.

    Returns *None* if the frame is missing energy or forces (unusable).
    """
    # --- energy -----------------------------------------------------------
    try:
        energy = atoms.get_potential_energy(force_consistent=True)
    except Exception:
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            log.warning("  ↳ No energy found in frame from %s — skipping.", source)
            return None

    # --- forces -----------------------------------------------------------
    try:
        forces = atoms.get_forces()
    except Exception:
        log.warning("  ↳ No forces found in frame from %s — skipping.", source)
        return None

    # --- stress / virial --------------------------------------------------
    virial = None
    try:
        # ASE returns Voigt stress in eV/Å³ (already converted from VASP kBar)
        stress_voigt = atoms.get_stress()  # shape (6,)  eV/Å³
        vol = atoms.get_volume()
        # Convert Voigt stress → full 3×3 virial = -stress * volume
        # Voigt order: xx, yy, zz, yz, xz, xy
        s = stress_voigt
        stress_3x3 = np.array([
            [s[0], s[5], s[4]],
            [s[5], s[1], s[3]],
            [s[4], s[3], s[2]],
        ])
        if convert_stress:
            # If the Atoms object already carries VASP-native kBar stress,
            # multiply by the conversion factor; otherwise ASE has done it.
            stress_3x3 = stress_3x3 * KBAR_TO_EV_ANG3
        virial = -stress_3x3 * vol  # eV  (gracemaker virial convention)
    except Exception:
        pass  # stress is optional

    # --- build tagged Atoms object ----------------------------------------
    new_atoms = atoms.copy()
    new_atoms.info.clear()

    new_atoms.info[energy_key] = energy
    new_atoms.info["config_type"] = config_type
    new_atoms.info["source_file"] = source

    new_atoms.arrays[forces_key] = forces.copy()

    if virial is not None:
        new_atoms.info[stress_key] = virial

    return new_atoms


def read_vasp_file(
    path: Path,
    selection: str,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    config_type: str,
    convert_stress: bool,
) -> List[Atoms]:
    """Read a single VASP OUTCAR or vasprun.xml and return tagged frames."""
    log.info("Reading: %s", path)
    fmt = "vasp-out" if path.name.upper() == "OUTCAR" else "vasp-xml"
    try:
        frames = read(str(path), index=":", format=fmt)
        if isinstance(frames, Atoms):
            frames = [frames]
    except Exception as exc:
        log.error("  ↳ Failed to read %s: %s", path, exc)
        return []

    log.info("  ↳ Found %d ionic steps", len(frames))
    frames = _apply_selection(frames, selection)
    log.info("  ↳ Keeping %d frame(s) after '%s' selection", len(frames), selection)

    tagged: List[Atoms] = []
    for atoms in frames:
        result = _tag_frame(
            atoms, str(path), energy_key, forces_key, stress_key,
            config_type, convert_stress,
        )
        if result is not None:
            tagged.append(result)

    return tagged


def walk_vasp_tree(
    root: Path,
    pattern: str,
) -> List[Path]:
    """Recursively find all OUTCAR or vasprun.xml files under *root*."""
    found: List[Path] = []
    for name in ("OUTCAR", "vasprun.xml"):
        if pattern in ("auto", name.upper()):
            found.extend(root.rglob(name))
    found.sort()
    log.info("Found %d VASP output file(s) under '%s'", len(found), root)
    return found


# ── Main function ─────────────────────────────────────────────────────────────

def convert(
    input_path: Optional[Path],
    input_dir: Optional[Path],
    output: Path,
    selection: str,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    config_type: str,
    convert_stress: bool,
    file_type: str,
    verbose: bool,
) -> int:
    """
    Main conversion routine.

    Returns the total number of frames written.
    """
    if verbose:
        log.setLevel(logging.DEBUG)

    all_frames: List[Atoms] = []

    # --- collect source files ---------------------------------------------
    sources: List[Path] = []
    if input_path is not None:
        sources = [input_path]
    elif input_dir is not None:
        sources = walk_vasp_tree(input_dir, file_type.upper())
    else:
        log.error("Provide --input or --input-dir.")
        return 0

    if not sources:
        log.error("No VASP output files found. Exiting.")
        return 0

    # --- process each file ------------------------------------------------
    for path in sources:
        frames = read_vasp_file(
            path, selection, energy_key, forces_key, stress_key,
            config_type, convert_stress,
        )
        all_frames.extend(frames)

    if not all_frames:
        log.error("No usable frames extracted. Check your VASP outputs.")
        return 0

    log.info("Total frames to write: %d", len(all_frames))

    # --- write extxyz -----------------------------------------------------
    output.parent.mkdir(parents=True, exist_ok=True)
    write(str(output), all_frames, format="extxyz")
    log.info("Written: %s", output.resolve())

    # --- summary statistics -----------------------------------------------
    energies = np.array([a.info.get(energy_key, np.nan) for a in all_frames])
    n_atoms  = np.array([len(a) for a in all_frames])
    e_per_atom = energies / n_atoms
    log.info(
        "Energy/atom  |  mean = %.4f  std = %.4f  min = %.4f  max = %.4f  (eV/atom)",
        np.nanmean(e_per_atom), np.nanstd(e_per_atom),
        np.nanmin(e_per_atom), np.nanmax(e_per_atom),
    )

    n_with_stress = sum(1 for a in all_frames if stress_key in a.info)
    log.info("Frames with stress: %d / %d", n_with_stress, len(all_frames))

    return len(all_frames)


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--input", type=Path, metavar="FILE",
        help="Single OUTCAR or vasprun.xml file.",
    )
    src.add_argument(
        "--input-dir", type=Path, metavar="DIR",
        help="Root directory; recursively searches for OUTCAR / vasprun.xml.",
    )

    p.add_argument(
        "--output", type=Path, default=Path("dataset.extxyz"),
        help="Output extended XYZ file. (default: dataset.extxyz)",
    )
    p.add_argument(
        "--selection", choices=["all", "first", "last", "first_and_last"],
        default="all",
        help="Which frames to keep from each VASP run. (default: all)",
    )
    p.add_argument(
        "--file-type", choices=["auto", "OUTCAR", "vasprun.xml"],
        default="auto",
        help="Which type of VASP output to search for when using --input-dir. "
             "(default: auto — finds both)",
    )
    p.add_argument(
        "--energy-key", default="REF_energy",
        help="Key for total energy in the extxyz info dict. (default: REF_energy)",
    )
    p.add_argument(
        "--forces-key", default="REF_forces",
        help="Key for forces in the extxyz arrays dict. (default: REF_forces)",
    )
    p.add_argument(
        "--stress-key", default="REF_virial",
        help="Key for virial tensor in the extxyz info dict. (default: REF_virial)",
    )
    p.add_argument(
        "--config-type", default="default",
        help="Value of 'config_type' tag written to each frame. (default: default)",
    )
    p.add_argument(
        "--convert-stress", action="store_true", default=False,
        help="Apply kBar→eV/Å³ conversion manually (only needed if ASE did not "
             "convert the stress already — rare).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input is None and args.input_dir is None:
        parser.error("Provide either --input FILE or --input-dir DIR.")

    n = convert(
        input_path=args.input,
        input_dir=args.input_dir,
        output=args.output,
        selection=args.selection,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        config_type=args.config_type,
        convert_stress=args.convert_stress,
        file_type=args.file_type,
        verbose=args.verbose,
    )
    if n == 0:
        sys.exit(1)
    log.info("Done. %d frame(s) written.", n)


if __name__ == "__main__":
    main()
