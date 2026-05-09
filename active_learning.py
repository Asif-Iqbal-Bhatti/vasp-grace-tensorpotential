#!/usr/bin/env python3
"""
active_learning.py

Committee-based Active Learning and Uncertainty Quantification for GRACE models.

Designed for finetuning workflows on low-symmetry structures (e.g., grain boundaries).
Loads N GRACE models trained on the same system, measures disagreement in predicted
energies and forces as a proxy for model uncertainty, and flags high-uncertainty
structures for DFT labelling.

Usage (single POSCAR):
    python active_learning.py --poscar POSCAR --models m1.pb m2.pb m3.pb

Usage (screen an MD trajectory):
    python active_learning.py --xdatcar XDATCAR --models m1.pb m2.pb m3.pb --stride 10

Usage (screen a directory of POSCARs):
    python active_learning.py --poscar_dir ./structures --models m1.pb m2.pb m3.pb

Outputs:
    uncertainty_log.dat         per-structure uncertainty metrics
    flagged/POSCAR_uncertain_N  structures above the threshold (ready for DFT)
"""

import os
import sys
import argparse
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

from ase.io import read, write


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def _load_single_model(model_path):
    from tensorpotential.calculator.foundation_models import grace_fm
    from tensorpotential.calculator import TPCalculator

    if os.path.exists(model_path):
        print(f"  Loading custom model: {model_path}")
        return TPCalculator(model_path)
    else:
        print(f"  Loading foundation model: {model_path}")
        return grace_fm(model_path)


# ──────────────────────────────────────────────
# Committee model
# ──────────────────────────────────────────────

class CommitteeModel:
    """
    Wraps N GRACE calculators and quantifies uncertainty via committee disagreement.

    Uncertainty metrics:
        std_energy          : std of total energies across committee (eV)
        max_force_unc       : max per-atom force-magnitude std across committee (eV/Å)
        mean_force_unc      : mean per-atom force-magnitude std across committee (eV/Å)
        per_atom_force_unc  : (natoms,) array of per-atom force uncertainty
    """

    def __init__(self, model_paths):
        if len(model_paths) < 2:
            raise ValueError(
                "Committee uncertainty requires at least 2 models. "
                "Train the same architecture on different data subsets or random seeds."
            )
        print(f"Loading {len(model_paths)} committee members...")
        self.calcs = [_load_single_model(p) for p in model_paths]
        self.n = len(self.calcs)
        print(f"Committee ready ({self.n} models).\n")

    def evaluate(self, atoms):
        """
        Run all committee members on `atoms` and return uncertainty metrics.

        Returns a dict with keys:
            mean_energy, std_energy,
            mean_forces, max_force_unc, mean_force_unc, per_atom_force_unc
        """
        energies = np.empty(self.n)
        forces_stack = np.empty((self.n, len(atoms), 3))

        for i, calc in enumerate(self.calcs):
            a = atoms.copy()
            a.calc = calc
            energies[i] = a.get_potential_energy()
            forces_stack[i] = a.get_forces()

        mean_energy = energies.mean()
        std_energy = energies.std()

        mean_forces = forces_stack.mean(axis=0)          # (natoms, 3)
        force_norms = np.linalg.norm(forces_stack, axis=2)  # (n, natoms)
        per_atom_unc = force_norms.std(axis=0)           # (natoms,)

        return {
            "mean_energy": mean_energy,
            "std_energy": std_energy,
            "mean_forces": mean_forces,
            "max_force_unc": per_atom_unc.max(),
            "mean_force_unc": per_atom_unc.mean(),
            "per_atom_force_unc": per_atom_unc,
        }


# ──────────────────────────────────────────────
# Screening helpers
# ──────────────────────────────────────────────

def screen_structures(structures, committee, threshold, output_dir="flagged"):
    """
    Evaluate uncertainty for each structure in `structures`.

    Saves flagged structures (max_force_unc > threshold) to output_dir
    and returns a list of result dicts for logging.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    flagged_count = 0

    for idx, atoms in enumerate(structures):
        print(f"  Evaluating structure {idx + 1}/{len(structures)}...", end=" ", flush=True)
        metrics = committee.evaluate(atoms)

        flagged = metrics["max_force_unc"] > threshold
        if flagged:
            out_path = os.path.join(output_dir, f"POSCAR_uncertain_{flagged_count:04d}")
            write(out_path, atoms, format="vasp")
            flagged_count += 1

        results.append({
            "index": idx,
            "mean_energy": metrics["mean_energy"],
            "std_energy": metrics["std_energy"],
            "max_force_unc": metrics["max_force_unc"],
            "mean_force_unc": metrics["mean_force_unc"],
            "flagged": flagged,
        })

        status = "FLAGGED" if flagged else "ok"
        print(
            f"E={metrics['mean_energy']:10.4f} eV  "
            f"σ_E={metrics['std_energy']:.4f} eV  "
            f"max_σ_F={metrics['max_force_unc']:.4f} eV/Å  [{status}]"
        )

    print(f"\n{flagged_count}/{len(structures)} structures flagged (threshold={threshold} eV/Å).")
    if flagged_count:
        print(f"Flagged POSCARs written to: {output_dir}/")

    return results


def write_uncertainty_log(results, filename="uncertainty_log.dat"):
    """Write per-structure uncertainty metrics to a plain-text file."""
    with open(filename, "w") as f:
        f.write(
            f"{'#idx':>6}  {'mean_E(eV)':>14}  {'std_E(eV)':>12}  "
            f"{'max_σ_F(eV/Å)':>14}  {'mean_σ_F(eV/Å)':>15}  {'flagged':>8}\n"
        )
        for r in results:
            f.write(
                f"{r['index']:6d}  {r['mean_energy']:14.6f}  {r['std_energy']:12.6f}  "
                f"{r['max_force_unc']:14.6f}  {r['mean_force_unc']:15.6f}  "
                f"{'YES' if r['flagged'] else 'no':>8}\n"
            )
    print(f"Uncertainty log written to: {filename}")


# ──────────────────────────────────────────────
# Per-atom uncertainty map (grain boundary analysis)
# ──────────────────────────────────────────────

def write_per_atom_uncertainty(atoms, per_atom_unc, filename="per_atom_uncertainty.dat"):
    """
    Write per-atom force uncertainty alongside fractional coordinates.
    Useful for identifying which atoms (e.g. at a grain boundary interface)
    are uncertain and should be targeted for DFT labelling.
    """
    scaled = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()

    with open(filename, "w") as f:
        f.write(f"# {'atom':>5}  {'elem':>4}  {'sx':>10}  {'sy':>10}  {'sz':>10}  {'σ_F(eV/Å)':>12}\n")
        for i, (sym, pos, unc) in enumerate(zip(symbols, scaled, per_atom_unc)):
            f.write(f"{i:6d}  {sym:>4}  {pos[0]:10.6f}  {pos[1]:10.6f}  {pos[2]:10.6f}  {unc:12.6f}\n")
    print(f"Per-atom uncertainty written to: {filename}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Committee-based uncertainty quantification for GRACE MLIP. "
            "Flags high-uncertainty structures for DFT labelling."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--poscar", help="Single POSCAR/CONTCAR file to evaluate.")
    src.add_argument("--xdatcar", help="XDATCAR trajectory to screen.")
    src.add_argument("--poscar_dir", help="Directory of POSCAR files to screen.")

    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to GRACE model files (≥2 required for committee UQ)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Max per-atom force uncertainty threshold in eV/Å (default: 0.1)."
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Read every Nth frame from XDATCAR (default: 1 = all frames)."
    )
    parser.add_argument(
        "--output_dir", default="flagged",
        help="Directory to write flagged POSCAR files (default: flagged/)."
    )
    parser.add_argument(
        "--log", default="uncertainty_log.dat",
        help="Output log filename (default: uncertainty_log.dat)."
    )
    parser.add_argument(
        "--per_atom", action="store_true",
        help="Also write per-atom uncertainty for the first (or only) structure."
    )
    args = parser.parse_args()

    # Load structures
    if args.poscar:
        if not os.path.exists(args.poscar):
            print(f"Error: {args.poscar} not found."); sys.exit(1)
        structures = [read(args.poscar, format="vasp")]
        print(f"Loaded 1 structure from {args.poscar}")

    elif args.xdatcar:
        if not os.path.exists(args.xdatcar):
            print(f"Error: {args.xdatcar} not found."); sys.exit(1)
        all_frames = read(args.xdatcar, index=":", format="vasp-xdatcar")
        structures = all_frames[:: args.stride]
        print(f"Loaded {len(structures)} frames from {args.xdatcar} (stride={args.stride})")

    else:  # poscar_dir
        if not os.path.isdir(args.poscar_dir):
            print(f"Error: {args.poscar_dir} is not a directory."); sys.exit(1)
        poscar_files = sorted(
            os.path.join(args.poscar_dir, f)
            for f in os.listdir(args.poscar_dir)
            if "POSCAR" in f or "CONTCAR" in f
        )
        if not poscar_files:
            print(f"No POSCAR/CONTCAR files found in {args.poscar_dir}."); sys.exit(1)
        structures = [read(p, format="vasp") for p in poscar_files]
        print(f"Loaded {len(structures)} structures from {args.poscar_dir}/")

    # Build committee
    committee = CommitteeModel(args.models)

    # Screen
    print(f"Screening {len(structures)} structure(s) — threshold = {args.threshold} eV/Å\n")
    results = screen_structures(structures, committee, args.threshold, args.output_dir)
    write_uncertainty_log(results, args.log)

    # Optional per-atom map for the first structure
    if args.per_atom:
        metrics = committee.evaluate(structures[0])
        write_per_atom_uncertainty(
            structures[0],
            metrics["per_atom_force_unc"],
            filename="per_atom_uncertainty.dat",
        )


if __name__ == "__main__":
    main()
