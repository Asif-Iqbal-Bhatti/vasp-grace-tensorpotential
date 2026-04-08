#!/usr/bin/env python3
"""
validate_grace_model.py
=======================
Evaluate a trained GRACE model against a reference dataset and produce:
  - Energy parity plot   (GRACE vs DFT, per atom)
  - Forces parity plot   (GRACE vs DFT, per component)
  - Stress parity plot   (GRACE vs DFT, per Voigt component)
  - Printed RMSE / MAE / MAX error statistics

Supports:
  - extxyz reference files
  - pkl.gz gracemaker dataset files

Usage
-----
  python validate_grace_model.py \\
      --model      seed/1/saved_model \\
      --test-data  test_set.extxyz \\
      --output-dir validation/ \\
      --energy-key REF_energy \\
      --forces-key REF_forces

Author: vasp-grace workflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.io import read

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_grace")


# ── Matplotlib setup (non-interactive) ───────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Data loading ──────────────────────────────────────────────────────────────

def load_reference_data(
    path: Path,
    energy_key: str,
    forces_key: str,
    stress_key: str,
) -> List[Atoms]:
    """Load reference structures from extxyz or pkl.gz."""
    suffix = path.suffix.lower()
    if suffix in (".xyz", ".extxyz") or (len(path.suffixes) >= 2 and path.suffixes[-1] == ".xyz"):
        frames = read(str(path), index=":")
        if isinstance(frames, Atoms):
            frames = [frames]
        log.info("Loaded %d frames from extxyz: %s", len(frames), path)
        return frames
    elif suffix in (".gz",) and ".pkl" in path.name:
        try:
            import pandas as pd
            df = pd.read_pickle(str(path))
            log.info("Loaded %d frames from pkl.gz: %s", len(df), path)
            # Convert dataframe rows to Atoms objects
            from ase import Atoms as _Atoms
            frames = []
            for _, row in df.iterrows():
                # gracemaker pkl format stores ASE Atoms-compatible dicts
                if hasattr(row, "ase_atoms"):
                    frames.append(row.ase_atoms)
            return frames
        except Exception as e:
            log.error("Could not load pkl.gz dataset: %s", e)
            sys.exit(1)
    else:
        # Try ASE generic read
        try:
            frames = read(str(path), index=":")
            if isinstance(frames, Atoms):
                frames = [frames]
            return frames
        except Exception as e:
            log.error("Unsupported file format: %s  (%s)", path, e)
            sys.exit(1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    frames: List[Atoms],
    calc,
    energy_key: str,
    forces_key: str,
    stress_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the GRACE model on all frames.

    Returns
    -------
    e_ref, e_pred   : energies per atom (eV/atom)
    f_ref, f_pred   : force components (eV/Å), shape (N_total*3,)
    s_ref, s_pred   : stress Voigt components (eV/Å³), shape (N_frames*6,)
    """
    e_ref_list, e_pred_list = [], []
    f_ref_list, f_pred_list = [], []
    s_ref_list, s_pred_list = [], []

    log.info("Evaluating %d frames...", len(frames))
    for i, atoms in enumerate(frames):
        n = len(atoms)

        # Reference values
        e_ref = atoms.info.get(energy_key)
        f_ref = atoms.arrays.get(forces_key)
        s_ref = atoms.info.get(stress_key)  # may be virial 3×3 or Voigt

        if e_ref is None or f_ref is None:
            log.debug("  Frame %d missing reference data — skipped.", i)
            continue

        # GRACE prediction
        atoms_pred = atoms.copy()
        atoms_pred.calc = calc
        try:
            e_pred = atoms_pred.get_potential_energy()
            f_pred = atoms_pred.get_forces()
        except Exception as exc:
            log.warning("  Frame %d: calculator error — %s", i, exc)
            continue

        e_ref_list.append(e_ref / n)
        e_pred_list.append(e_pred / n)

        f_ref_list.extend(f_ref.flatten().tolist())
        f_pred_list.extend(f_pred.flatten().tolist())

        # Stress
        if s_ref is not None:
            try:
                s_pred_voigt = atoms_pred.get_stress()  # eV/Å³
                s_ref_arr = np.array(s_ref)
                if s_ref_arr.shape == (3, 3):
                    # Convert virial 3×3 → Voigt eV/Å³
                    vol = atoms.get_volume()
                    s_ref_voigt = np.array([
                        -s_ref_arr[0, 0], -s_ref_arr[1, 1], -s_ref_arr[2, 2],
                        -s_ref_arr[1, 2], -s_ref_arr[0, 2], -s_ref_arr[0, 1],
                    ]) / vol
                else:
                    s_ref_voigt = s_ref_arr.flatten()[:6]
                s_ref_list.extend(s_ref_voigt.tolist())
                s_pred_list.extend(s_pred_voigt.tolist())
            except Exception:
                pass

        if (i + 1) % max(1, len(frames) // 10) == 0:
            log.info("  Evaluated %d / %d frames.", i + 1, len(frames))

    return (
        np.array(e_ref_list), np.array(e_pred_list),
        np.array(f_ref_list), np.array(f_pred_list),
        np.array(s_ref_list), np.array(s_pred_list),
    )


# ── Statistics ────────────────────────────────────────────────────────────────

def print_stats(label: str, ref: np.ndarray, pred: np.ndarray, unit: str) -> dict:
    """Compute and print RMSE / MAE / MAX error statistics."""
    if len(ref) == 0:
        log.warning("%s: no data to evaluate.", label)
        return {}
    diff = pred - ref
    rmse = np.sqrt(np.mean(diff**2))
    mae  = np.mean(np.abs(diff))
    max_ = np.max(np.abs(diff))
    r2   = 1 - np.sum(diff**2) / np.sum((ref - ref.mean())**2) if ref.std() > 0 else float("nan")
    log.info("  %-18s  RMSE=%9.5f  MAE=%9.5f  MAX=%9.5f  R²=%.4f   [%s]",
             label, rmse, mae, max_, r2, unit)
    return {"rmse": rmse, "mae": mae, "max": max_, "r2": r2}


# ── Plotting ──────────────────────────────────────────────────────────────────

def parity_plot(
    ref: np.ndarray,
    pred: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    unit: str,
    max_points: int = 20_000,
) -> None:
    """Scatter parity plot with identity line, saved to *out_path*."""
    if len(ref) == 0:
        log.warning("No data for plot: %s", title)
        return

    # Subsample for clarity
    if len(ref) > max_points:
        idx = np.random.choice(len(ref), max_points, replace=False)
        ref  = ref[idx]
        pred = pred[idx]

    diff  = pred - ref
    rmse  = np.sqrt(np.mean(diff**2))
    mae   = np.mean(np.abs(diff))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Parity plot
    ax = axes[0]
    ax.scatter(ref, pred, alpha=0.3, s=4, color="#2166ac", rasterized=True)
    lim = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
    ax.plot(lim, lim, "r--", lw=1.2, label="ideal")
    ax.set_xlabel(f"DFT {unit}")
    ax.set_ylabel(f"GRACE {unit}")
    ax.set_title(title)
    ax.legend()
    ax.text(
        0.05, 0.95,
        f"RMSE = {rmse:.4f} {unit}\nMAE  = {mae:.4f} {unit}\nN = {len(ref):,}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    # Error histogram
    ax = axes[1]
    ax.hist(diff, bins=60, color="#4dac26", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", lw=1.2, ls="--")
    ax.set_xlabel(f"GRACE − DFT  [{unit}]")
    ax.set_ylabel("Count")
    ax.set_title(f"{title} — error distribution")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="GRACE saved_model directory or YAML file.")
    parser.add_argument("--model-type", choices=["auto", "grace", "grace-fs"],
                        default="auto")
    parser.add_argument("--active-set", default=None, metavar="FILE")
    parser.add_argument("--test-data", required=True, metavar="FILE",
                        help="Test dataset (extxyz or pkl.gz).")
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument("--forces-key", default="REF_forces")
    parser.add_argument("--stress-key", default="REF_virial")
    parser.add_argument("--output-dir", type=Path, default=Path("validation"))
    parser.add_argument("--max-plot-points", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    np.random.seed(args.seed)

    # ── Load reference data ───────────────────────────────────────────────────
    frames = load_reference_data(
        Path(args.test_data), args.energy_key, args.forces_key, args.stress_key,
    )
    if not frames:
        log.error("No frames loaded from %s.", args.test_data)
        sys.exit(1)

    # ── Load GRACE calculator ─────────────────────────────────────────────────
    # Re-use loader from grace_ase_calculator
    sys.path.insert(0, str(Path(__file__).parent))
    from grace_ase_calculator import load_grace_calculator
    calc = load_grace_calculator(args.model, args.model_type, args.active_set, "gpu")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    e_ref, e_pred, f_ref, f_pred, s_ref, s_pred = evaluate(
        frames, calc, args.energy_key, args.forces_key, args.stress_key,
    )

    log.info("")
    log.info("━━━━ Error Statistics ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    stats = {}
    stats["energy"] = print_stats("Energy/atom",  e_ref, e_pred, "eV/atom")
    stats["forces"] = print_stats("Forces",        f_ref, f_pred, "eV/Å")
    if len(s_ref):
        stats["stress"] = print_stats("Stress",   s_ref, s_pred, "eV/Å³")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ── Save statistics to file ───────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "error_statistics.txt"
    with open(stats_path, "w") as fh:
        fh.write("GRACE Model Validation\n")
        fh.write(f"Model:      {args.model}\n")
        fh.write(f"Test data:  {args.test_data}\n")
        fh.write(f"N frames:   {len(frames)}\n\n")
        for name, s in stats.items():
            if s:
                fh.write(f"[{name}]\n")
                for k, v in s.items():
                    fh.write(f"  {k:6s} = {v:.6f}\n")
                fh.write("\n")
    log.info("Statistics saved: %s", stats_path)

    # ── Parity plots ──────────────────────────────────────────────────────────
    log.info("Generating parity plots...")
    parity_plot(e_ref, e_pred,
                "DFT Energy (eV/atom)", "GRACE Energy (eV/atom)",
                "Energy parity", args.output_dir / "parity_energy.png",
                "eV/atom", args.max_plot_points)

    parity_plot(f_ref, f_pred,
                "DFT Forces (eV/Å)", "GRACE Forces (eV/Å)",
                "Forces parity", args.output_dir / "parity_forces.png",
                "eV/Å", args.max_plot_points)

    if len(s_ref):
        parity_plot(s_ref, s_pred,
                    "DFT Stress (eV/Å³)", "GRACE Stress (eV/Å³)",
                    "Stress parity", args.output_dir / "parity_stress.png",
                    "eV/Å³", args.max_plot_points)

    log.info("Validation complete. Output in: %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
