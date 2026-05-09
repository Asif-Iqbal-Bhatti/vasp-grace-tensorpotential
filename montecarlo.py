#!/usr/bin/env python3
"""
montecarlo.py

Metropolis Monte Carlo structural sampling using GRACE MLIP energies.

Designed for exploring asymmetric/low-symmetry structures such as grain boundaries.
Supports displacement moves, atom-swap moves (useful for Li-site disorder in
Li6PS5Cl), and optional live uncertainty flagging via a committee of GRACE models.

Usage (basic):
    python montecarlo.py --poscar POSCAR --model GRACE-2L-OAM --temperature 800 --steps 50000

Usage (with committee UQ — flags uncertain snapshots for DFT):
    python montecarlo.py --poscar POSCAR --models m1.pb m2.pb m3.pb --temperature 800

Usage (swap moves for Li-site disorder):
    python montecarlo.py --poscar POSCAR --model m1.pb --swap Li S --temperature 1000

Outputs:
    MC_energies.dat         step, energy, accepted, [uncertainty if committee]
    XDATCAR_MC              trajectory in VASP format
    flagged/POSCAR_MC_N     high-uncertainty snapshots for DFT (committee mode only)
"""

import os
import sys
import argparse
import numpy as np
from itertools import groupby

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

from ase.io import read, write
from ase import units


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def _load_model(model_path):
    from tensorpotential.calculator.foundation_models import grace_fm
    from tensorpotential.calculator import TPCalculator

    if os.path.exists(model_path):
        return TPCalculator(model_path)
    else:
        return grace_fm(model_path)


# ──────────────────────────────────────────────
# Metropolis MC engine
# ──────────────────────────────────────────────

class MetropolisMC:
    """
    Metropolis Monte Carlo sampler using a GRACE MLIP as the energy engine.

    Move types
    ----------
    displacement : randomly displace one atom by up to `max_disp` Å
    swap         : swap positions of two atoms of different species
                   (useful for cation disorder in Li6PS5Cl grain boundaries)

    Optional committee UQ
    ---------------------
    Pass a CommitteeModel instance to `committee` and a threshold to
    `uq_threshold` to flag high-uncertainty snapshots during the run.
    These are saved to `flagged_dir` as individual POSCARs for DFT.
    """

    def __init__(
        self,
        atoms,
        calc,
        temperature,
        max_disp=0.15,
        swap_species=None,
        seed=42,
    ):
        self.atoms = atoms.copy()
        self.atoms.calc = calc
        self.temperature = temperature          # K
        self.kT = temperature * units.kB       # eV
        self.max_disp = max_disp               # Å
        self.swap_species = swap_species        # e.g. ["Li", "S"] or None
        self.rng = np.random.default_rng(seed)

        self.energy = self.atoms.get_potential_energy()
        self.n_accept = 0
        self.n_reject = 0
        self.step = 0

    # ── moves ──────────────────────────────────

    def _displacement_move(self):
        """Displace a random atom by a uniform random vector within max_disp."""
        idx = self.rng.integers(len(self.atoms))
        delta = (self.rng.random(3) - 0.5) * 2 * self.max_disp
        return idx, delta, None

    def _swap_move(self):
        """
        Swap positions of one atom of swap_species[0] with one of swap_species[1].
        Returns None if the species pair is not present.
        """
        symbols = np.array(self.atoms.get_chemical_symbols())
        idx_a = np.where(symbols == self.swap_species[0])[0]
        idx_b = np.where(symbols == self.swap_species[1])[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            return None
        a = self.rng.choice(idx_a)
        b = self.rng.choice(idx_b)
        return a, b

    # ── Metropolis step ─────────────────────────

    def _attempt_displacement(self):
        idx, delta, _ = self._displacement_move()
        old_pos = self.atoms.positions[idx].copy()

        self.atoms.positions[idx] += delta
        new_energy = self.atoms.get_potential_energy()
        dE = new_energy - self.energy

        if dE < 0 or self.rng.random() < np.exp(-dE / self.kT):
            self.energy = new_energy
            return True
        else:
            self.atoms.positions[idx] = old_pos
            _ = self.atoms.get_potential_energy()  # reset calculator cache
            return False

    def _attempt_swap(self):
        result = self._swap_move()
        if result is None:
            return False

        a, b = result
        pos_a = self.atoms.positions[a].copy()
        pos_b = self.atoms.positions[b].copy()

        self.atoms.positions[a] = pos_b
        self.atoms.positions[b] = pos_a
        new_energy = self.atoms.get_potential_energy()
        dE = new_energy - self.energy

        if dE < 0 or self.rng.random() < np.exp(-dE / self.kT):
            self.energy = new_energy
            return True
        else:
            self.atoms.positions[a] = pos_a
            self.atoms.positions[b] = pos_b
            _ = self.atoms.get_potential_energy()
            return False

    # ── main run ───────────────────────────────

    def run(
        self,
        steps,
        swap_prob=0.0,
        save_interval=100,
        committee=None,
        uq_threshold=0.1,
        flagged_dir="flagged",
        log_file="MC_energies.dat",
        xdatcar="XDATCAR_MC",
    ):
        """
        Run the MC loop.

        Parameters
        ----------
        steps           : total MC steps
        swap_prob       : probability of attempting a swap move (0.0–1.0)
        save_interval   : write trajectory frame every N steps
        committee       : CommitteeModel instance for live UQ (optional)
        uq_threshold    : flag structures with max_force_unc > threshold (eV/Å)
        flagged_dir     : directory for uncertain POSCARs
        log_file        : energy + uncertainty log
        xdatcar         : VASP-format trajectory output
        """
        use_uq = committee is not None
        if use_uq:
            os.makedirs(flagged_dir, exist_ok=True)

        self._init_xdatcar(xdatcar)
        flagged_count = 0

        header = (
            f"{'#step':>8}  {'energy(eV)':>14}  {'dE(eV)':>12}  "
            f"{'accepted':>9}  {'acc_rate':>9}"
        )
        if use_uq:
            header += f"  {'max_σ_F(eV/Å)':>14}  {'uq_flagged':>10}"

        with open(log_file, "w") as f:
            f.write(header + "\n")

        print(f"\n--- Metropolis MC ---")
        print(f"Temperature : {self.temperature} K")
        print(f"Steps       : {steps}")
        print(f"Max disp    : {self.max_disp} Å")
        if self.swap_species:
            print(f"Swap moves  : {self.swap_species[0]} ↔ {self.swap_species[1]}  (prob={swap_prob:.2f})")
        print(f"UQ active   : {use_uq}" + (f"  (threshold={uq_threshold} eV/Å)" if use_uq else ""))
        print(f"Save every  : {save_interval} steps\n")

        e0 = self.energy

        for step in range(1, steps + 1):
            self.step = step

            # Choose move type
            if self.swap_species and self.rng.random() < swap_prob:
                accepted = self._attempt_swap()
            else:
                accepted = self._attempt_displacement()

            if accepted:
                self.n_accept += 1
            else:
                self.n_reject += 1

            acc_rate = self.n_accept / step

            # UQ evaluation at save points
            max_unc = float("nan")
            uq_flagged = False
            if step % save_interval == 0:
                self._append_xdatcar(xdatcar, step)

                if use_uq:
                    metrics = committee.evaluate(self.atoms)
                    max_unc = metrics["max_force_unc"]
                    uq_flagged = max_unc > uq_threshold
                    if uq_flagged:
                        out = os.path.join(flagged_dir, f"POSCAR_MC_{flagged_count:05d}")
                        write(out, self.atoms, format="vasp")
                        flagged_count += 1

            # Log
            if step % save_interval == 0 or step == 1:
                dE = self.energy - e0
                row = (
                    f"{step:8d}  {self.energy:14.6f}  {dE:12.6f}  "
                    f"{'yes' if accepted else 'no':>9}  {acc_rate:9.4f}"
                )
                if use_uq:
                    flag_str = "FLAGGED" if uq_flagged else "ok"
                    unc_str = f"{max_unc:.4f}" if not np.isnan(max_unc) else "  ---   "
                    row += f"  {unc_str:>14}  {flag_str:>10}"
                with open(log_file, "a") as f:
                    f.write(row + "\n")

                print(
                    f"  step {step:6d}/{steps}  E={self.energy:10.4f} eV  "
                    f"acc={acc_rate:.3f}"
                    + (f"  max_σ_F={max_unc:.4f}" if use_uq and not np.isnan(max_unc) else "")
                    + (f"  [FLAGGED]" if uq_flagged else "")
                )

        print(f"\nMC finished.")
        print(f"  Accepted : {self.n_accept}/{steps}  ({100*self.n_accept/steps:.1f}%)")
        if use_uq:
            print(f"  Flagged  : {flagged_count} structures written to {flagged_dir}/")
        print(f"  Log      : {log_file}")
        print(f"  Traj     : {xdatcar}")

    # ── XDATCAR helpers ────────────────────────

    def _init_xdatcar(self, filename):
        symbols = self.atoms.get_chemical_symbols()
        grouped = [(k, len(list(g))) for k, g in groupby(symbols)]
        with open(filename, "w") as f:
            f.write("MC trajectory from vasp-grace-tensorpotential\n")
            f.write("   1.0\n")
            for row in self.atoms.get_cell():
                f.write(f"    {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}\n")
            f.write("   " + "   ".join(g[0] for g in grouped) + "\n")
            f.write("   " + "   ".join(str(g[1]) for g in grouped) + "\n")

    def _append_xdatcar(self, filename, step):
        scaled = self.atoms.get_scaled_positions(wrap=True)
        with open(filename, "a") as f:
            f.write(f"Direct configuration={step:8d}\n")
            for pos in scaled:
                f.write(f"  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Metropolis Monte Carlo sampling with GRACE MLIP. "
            "Optionally integrates committee UQ to flag uncertain structures."
        )
    )
    parser.add_argument("--poscar", default="POSCAR", help="Input structure (POSCAR/CONTCAR).")

    model_grp = parser.add_mutually_exclusive_group(required=True)
    model_grp.add_argument("--model", help="Single GRACE model path or foundation model name.")
    model_grp.add_argument(
        "--models", nargs="+",
        help="≥2 GRACE models for committee UQ (enables live uncertainty flagging)."
    )

    parser.add_argument("--temperature", type=float, default=500.0, help="Temperature in K (default: 500).")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MC steps (default: 10000).")
    parser.add_argument("--max_disp", type=float, default=0.15, help="Max atomic displacement in Å (default: 0.15).")
    parser.add_argument(
        "--swap", nargs=2, metavar=("ELEM_A", "ELEM_B"),
        help="Enable swap moves between two element types, e.g. --swap Li S"
    )
    parser.add_argument("--swap_prob", type=float, default=0.2,
                        help="Fraction of moves that are swaps when --swap is set (default: 0.2).")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save trajectory frame every N steps (default: 100).")
    parser.add_argument("--uq_threshold", type=float, default=0.1,
                        help="Force uncertainty threshold for flagging in eV/Å (default: 0.1).")
    parser.add_argument("--flagged_dir", default="flagged",
                        help="Directory for uncertain POSCARs (default: flagged/).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--log", default="MC_energies.dat", help="Energy log filename.")
    parser.add_argument("--xdatcar", default="XDATCAR_MC", help="Trajectory output filename.")

    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    atoms = read(args.poscar, format="vasp")
    print(f"Loaded structure: {len(atoms)} atoms from {args.poscar}")

    # Single model
    committee = None
    if args.model:
        calc = _load_model(args.model)
        atoms.calc = calc
    else:
        # Committee: use first model as the MC energy engine, full committee for UQ
        from active_learning import CommitteeModel
        committee = CommitteeModel(args.models)
        calc = committee.calcs[0]
        atoms.calc = calc
        print(f"Using model[0] for MC energy; full committee ({len(args.models)}) for UQ.\n")

    mc = MetropolisMC(
        atoms=atoms,
        calc=calc,
        temperature=args.temperature,
        max_disp=args.max_disp,
        swap_species=args.swap if args.swap else None,
        seed=args.seed,
    )

    mc.run(
        steps=args.steps,
        swap_prob=args.swap_prob if args.swap else 0.0,
        save_interval=args.save_interval,
        committee=committee,
        uq_threshold=args.uq_threshold,
        flagged_dir=args.flagged_dir,
        log_file=args.log,
        xdatcar=args.xdatcar,
    )

    # Save final structure
    write("CONTCAR_MC", mc.atoms, format="vasp")
    print("Final structure written to CONTCAR_MC")


if __name__ == "__main__":
    main()
