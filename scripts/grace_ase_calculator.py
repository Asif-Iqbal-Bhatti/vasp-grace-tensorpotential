#!/usr/bin/env python3
"""
grace_ase_calculator.py
=======================
Run ASE-based tasks (structure relaxation, NVT/NPT molecular dynamics,
single-point evaluation) using a trained GRACE interatomic potential.

Supports:
  - GRACE-2L / GRACE-1L  →  GRACECalculator  (tensorpotential)
  - GRACE/FS (YAML)       →  PyGRACEFSCalculator  (python-ace)
  - Foundation models     →  same interfaces, downloaded via grace_models

Tasks
-----
  relax   – BFGS or FIRE geometry optimisation
  npt     – NPT (isothermal-isobaric) molecular dynamics via ASE
  nvt     – NVT (Nosé-Hoover) molecular dynamics
  nve     – NVE (microcanonical) molecular dynamics
  sp      – single-point energy/forces/stress

Usage
-----
  python grace_ase_calculator.py \\
      --structure  POSCAR \\
      --model      seed/1/saved_model \\
      --task       relax \\
      --fmax       0.01 \\
      --output     relaxed.extxyz

  python grace_ase_calculator.py \\
      --structure  POSCAR \\
      --model      seed/1/saved_model \\
      --task       nvt \\
      --temperature 1000 \\
      --steps      50000 \\
      --timestep   2.0 \\
      --output     nvt_traj.extxyz

  # GRACE/FS YAML model
  python grace_ase_calculator.py \\
      --structure  POSCAR \\
      --model      seed/1/FS_model.yaml \\
      --model-type grace-fs \\
      --active-set seed/1/FS_model.asi \\
      --task       npt

Author: vasp-grace workflow
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.optimize import BFGS, FIRE
from ase.constraints import UnitCellFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grace_ase")


# ── Calculator loader ─────────────────────────────────────────────────────────

def load_grace_calculator(
    model_path: str,
    model_type: str,
    active_set: Optional[str],
    device: str,
):
    """
    Load the appropriate GRACE ASE calculator.

    Parameters
    ----------
    model_path : str
        Path to the saved model directory (TF SavedModel) or YAML file (FS).
    model_type : str
        One of 'auto', 'grace', 'grace-fs'.
    active_set : str | None
        Path to *.asi* file for GRACE/FS extrapolation grading.
    device : str
        'cpu' or 'gpu' (GPU preferred for GRACE-1L/2L).
    """
    p = Path(model_path)

    # Auto-detect model type
    if model_type == "auto":
        if p.suffix in (".yaml", ".yml"):
            model_type = "grace-fs"
        else:
            model_type = "grace"

    if model_type == "grace-fs":
        # GRACE/FS via python-ace
        try:
            from pyace.asecalc import PyGRACEFSCalculator
        except ImportError as e:
            log.error(
                "python-ace is required for GRACE/FS models: pip install python-ace\n%s", e
            )
            sys.exit(1)

        calc = PyGRACEFSCalculator(str(p))
        if active_set:
            calc.set_active_set(active_set)
            log.info("Active set loaded: %s", active_set)
        log.info("Loaded GRACE/FS calculator from: %s", p)

    else:
        # GRACE-1L / GRACE-2L via tensorpotential
        try:
            from tensorpotential.calculator.grace import GRACECalculator
        except ImportError as e:
            log.error(
                "tensorpotential is required for GRACE models: pip install tensorpotential\n%s", e
            )
            sys.exit(1)

        calc = GRACECalculator(str(p))
        log.info("Loaded GRACE calculator from: %s", p)

    return calc


# ── Tasks ─────────────────────────────────────────────────────────────────────

def task_single_point(atoms: Atoms, output: Path) -> None:
    """Compute energy, forces, and stress for a single configuration."""
    log.info("Running single-point evaluation...")
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    try:
        s = atoms.get_stress()
        log.info("  Stress (Voigt, eV/Å³): %s", np.round(s, 6))
    except Exception:
        s = None

    log.info("  Energy            : %.6f eV", e)
    log.info("  Energy/atom       : %.6f eV/atom", e / len(atoms))
    log.info("  Max |force|       : %.6f eV/Å", np.max(np.linalg.norm(f, axis=1)))

    write(str(output), atoms, format="extxyz")
    log.info("Written: %s", output)


def task_relax(
    atoms: Atoms,
    output: Path,
    fmax: float,
    steps: int,
    relax_cell: bool,
    optimizer_name: str,
) -> None:
    """Geometry (and optionally cell) optimisation."""
    log.info("Running structure relaxation (fmax=%.4f eV/Å, relax_cell=%s)...",
             fmax, relax_cell)

    if relax_cell:
        opt_atoms = UnitCellFilter(atoms)
    else:
        opt_atoms = atoms

    Opt = FIRE if optimizer_name.upper() == "FIRE" else BFGS
    traj_path = output.with_suffix(".traj")

    opt = Opt(opt_atoms, trajectory=str(traj_path))

    t0 = time.time()
    converged = opt.run(fmax=fmax, steps=steps)
    elapsed = time.time() - t0

    e = atoms.get_potential_energy()
    f_max = np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    log.info("  Converged     : %s", converged)
    log.info("  Steps         : %d", opt.get_number_of_steps())
    log.info("  Energy        : %.6f eV  (%.6f eV/atom)", e, e / len(atoms))
    log.info("  Max |force|   : %.6f eV/Å", f_max)
    log.info("  Wall time     : %.1f s", elapsed)

    write(str(output), atoms, format="extxyz")
    log.info("Written: %s", output)


def _md_logger(atoms: Atoms, step: int, log_interval: int, t_K: float) -> None:
    e = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    t_inst = ekin / (1.5 * units.kB * len(atoms))
    log.info(
        "  step=%8d  E_pot=%12.4f eV  E_kin=%10.4f eV  T_inst=%.1f K",
        step, e, ekin, t_inst,
    )


def task_nvt(
    atoms: Atoms,
    output: Path,
    temperature: float,
    timestep_fs: float,
    steps: int,
    friction: float,
    log_interval: int,
    write_interval: int,
) -> None:
    """NVT (Langevin) molecular dynamics."""
    log.info(
        "Running NVT MD: T=%.0f K, dt=%.2f fs, steps=%d",
        temperature, timestep_fs, steps,
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature,
        friction=friction,
        logfile="-",
    )

    traj: list[Atoms] = []

    def step_callback():
        step = dyn.get_number_of_steps()
        if step % log_interval == 0:
            _md_logger(atoms, step, log_interval, temperature)
        if step % write_interval == 0:
            traj.append(atoms.copy())

    dyn.attach(step_callback, interval=1)

    t0 = time.time()
    dyn.run(steps)
    elapsed = time.time() - t0

    log.info("NVT MD complete. Wall time: %.1f s (%.2f ms/step)", elapsed, elapsed / steps * 1e3)
    write(str(output), traj, format="extxyz")
    log.info("Trajectory written: %s  (%d frames)", output, len(traj))


def task_npt(
    atoms: Atoms,
    output: Path,
    temperature: float,
    pressure_GPa: float,
    timestep_fs: float,
    steps: int,
    taut: float,
    taup: float,
    log_interval: int,
    write_interval: int,
) -> None:
    """NPT (Berendsen) molecular dynamics."""
    log.info(
        "Running NPT MD: T=%.0f K, P=%.2f GPa, dt=%.2f fs, steps=%d",
        temperature, pressure_GPa, timestep_fs, steps,
    )
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)

    dyn = NPTBerendsen(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature,
        pressure_au=pressure_GPa * units.GPa,
        taut=taut * units.fs,
        taup=taup * units.fs,
        compressibility_au=4.57e-5 / units.bar,
    )

    traj: list[Atoms] = []

    def step_callback():
        step = dyn.get_number_of_steps()
        if step % log_interval == 0:
            _md_logger(atoms, step, log_interval, temperature)
        if step % write_interval == 0:
            traj.append(atoms.copy())

    dyn.attach(step_callback, interval=1)

    t0 = time.time()
    dyn.run(steps)
    elapsed = time.time() - t0

    log.info("NPT MD complete. Wall time: %.1f s (%.2f ms/step)", elapsed, elapsed / steps * 1e3)
    write(str(output), traj, format="extxyz")
    log.info("Trajectory written: %s  (%d frames)", output, len(traj))


def task_nve(
    atoms: Atoms,
    output: Path,
    timestep_fs: float,
    steps: int,
    log_interval: int,
    write_interval: int,
) -> None:
    """NVE (microcanonical) molecular dynamics via velocity Verlet."""
    log.info("Running NVE MD: dt=%.2f fs, steps=%d", timestep_fs, steps)

    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    traj: list[Atoms] = []

    def step_callback():
        step = dyn.get_number_of_steps()
        if step % log_interval == 0:
            _md_logger(atoms, step, log_interval, 0.0)
        if step % write_interval == 0:
            traj.append(atoms.copy())

    dyn.attach(step_callback, interval=1)

    t0 = time.time()
    dyn.run(steps)
    elapsed = time.time() - t0

    log.info("NVE MD complete. Wall time: %.1f s (%.2f ms/step)", elapsed, elapsed / steps * 1e3)
    write(str(output), traj, format="extxyz")
    log.info("Trajectory written: %s  (%d frames)", output, len(traj))


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / model
    p.add_argument("--structure", required=True, metavar="FILE",
                   help="Input structure file (POSCAR, CIF, extxyz, …).")
    p.add_argument("--model", required=True, metavar="PATH",
                   help="Path to GRACE saved_model directory or YAML file.")
    p.add_argument("--model-type", choices=["auto", "grace", "grace-fs"],
                   default="auto",
                   help="Calculator backend. 'auto' detects from path. (default: auto)")
    p.add_argument("--active-set", default=None, metavar="FILE",
                   help="Path to *.asi active-set file for GRACE/FS extrapolation grading.")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                   help="Compute device. (default: gpu)")

    # Task
    p.add_argument("--task", required=True,
                   choices=["sp", "relax", "nvt", "npt", "nve"],
                   help="Task to run.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output file path. Defaults to <task>.extxyz")

    # Relaxation
    rel = p.add_argument_group("Relaxation (task=relax)")
    rel.add_argument("--fmax", type=float, default=0.02,
                     help="Max force convergence (eV/Å). (default: 0.02)")
    rel.add_argument("--relax-cell", action="store_true",
                     help="Allow unit cell to relax (UnitCellFilter).")
    rel.add_argument("--optimizer", choices=["BFGS", "FIRE"], default="BFGS",
                     help="ASE optimiser. (default: BFGS)")

    # MD shared
    md = p.add_argument_group("Molecular dynamics (task=nvt/npt/nve)")
    md.add_argument("--steps", type=int, default=10_000)
    md.add_argument("--timestep", type=float, default=2.0,
                    help="MD time step in fs. (default: 2.0)")
    md.add_argument("--temperature", type=float, default=300.0,
                    help="Target temperature in K. (default: 300)")
    md.add_argument("--pressure", type=float, default=0.0,
                    help="Target pressure in GPa for NPT. (default: 0.0)")
    md.add_argument("--friction", type=float, default=0.01,
                    help="Langevin friction coefficient (1/fs). (default: 0.01)")
    md.add_argument("--taut", type=float, default=100.0,
                    help="Berendsen temperature coupling time (fs). (default: 100)")
    md.add_argument("--taup", type=float, default=1000.0,
                    help="Berendsen pressure coupling time (fs). (default: 1000)")
    md.add_argument("--log-interval", type=int, default=100,
                    help="Print MD log every N steps. (default: 100)")
    md.add_argument("--write-interval", type=int, default=100,
                    help="Write frame to trajectory every N steps. (default: 100)")

    p.add_argument("--max-steps", type=int, default=10_000,
                   help="Max optimisation steps for relaxation. (default: 10000)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Default output filename
    output = args.output or Path(f"{args.task}.extxyz")

    # Load structure
    log.info("Loading structure: %s", args.structure)
    atoms = read(args.structure)
    log.info("  Atoms: %d  |  Formula: %s", len(atoms), atoms.get_chemical_formula())

    # Load GRACE calculator
    calc = load_grace_calculator(
        args.model, args.model_type, args.active_set, args.device,
    )
    atoms.calc = calc

    # Dispatch task
    if args.task == "sp":
        task_single_point(atoms, output)

    elif args.task == "relax":
        task_relax(
            atoms, output,
            fmax=args.fmax,
            steps=args.max_steps,
            relax_cell=args.relax_cell,
            optimizer_name=args.optimizer,
        )

    elif args.task == "nvt":
        task_nvt(
            atoms, output,
            temperature=args.temperature,
            timestep_fs=args.timestep,
            steps=args.steps,
            friction=args.friction,
            log_interval=args.log_interval,
            write_interval=args.write_interval,
        )

    elif args.task == "npt":
        task_npt(
            atoms, output,
            temperature=args.temperature,
            pressure_GPa=args.pressure,
            timestep_fs=args.timestep,
            steps=args.steps,
            taut=args.taut,
            taup=args.taup,
            log_interval=args.log_interval,
            write_interval=args.write_interval,
        )

    elif args.task == "nve":
        task_nve(
            atoms, output,
            timestep_fs=args.timestep,
            steps=args.steps,
            log_interval=args.log_interval,
            write_interval=args.write_interval,
        )


if __name__ == "__main__":
    main()
