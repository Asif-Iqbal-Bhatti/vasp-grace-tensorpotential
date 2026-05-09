#!/usr/bin/env python3
"""
thermal_conductivity.py

Lattice thermal conductivity via Reverse Non-Equilibrium MD (rNEMD).

Implements the Müller-Plathe (1997) velocity-exchange method:

    1. Divide the cell into N slabs along the z-axis.
    2. Every `swap_interval` steps swap the z-velocity of the atom with the
       highest kinetic energy in the cold slab (slab 0) with the atom with the
       lowest kinetic energy in the hot slab (slab N/2).
    3. This imposes a controlled heat flux J from cold→hot, and the system
       responds with a temperature gradient dT/dz (hot→cold).
    4. At steady state: κ = J / |dT/dz|

This method requires no per-atom stress tensor — it works with any force field
and is well-suited to GRACE MLIP.

Reference
---------
Müller-Plathe, F. J. Chem. Phys. 106, 6082 (1997).
https://doi.org/10.1063/1.473271

Usage
-----
    # Basic run at 300 K
    python thermal_conductivity.py --poscar POSCAR --model GRACE-2L-OAM \\
                                   --temperature 300 --steps 200000

    # Custom slab count and swap interval
    python thermal_conductivity.py --poscar POSCAR --model my_model.pb \\
                                   --temperature 500 --steps 300000 \\
                                   --nslabs 30 --swap_interval 50

Outputs
-------
    temperature_profile.dat   T(z) averaged over production phase
    kappa_convergence.dat     κ(t) running estimate during production
    kappa_summary.txt         final κ, gradient, flux, fit quality
    temperature_profile.png   plot of T(z) with linear fit
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation


# ──────────────────────────────────────────────────────────────────────────────
# Müller-Plathe rNEMD engine
# ──────────────────────────────────────────────────────────────────────────────

class MuellerPlatheNEMD:
    """
    Reverse NEMD thermal conductivity via velocity exchange.

    The cell is divided into slabs along the z-axis.
    Slab 0 (z ≈ 0)    → cold region (loses kinetic energy)
    Slab N/2 (z ≈ L/2) → hot region (gains kinetic energy)

    Heat current: J = ΔKE_total / (2 × t × A)
    (factor 2: heat flows in both ±z directions due to PBC)

    Thermal conductivity: κ = J / |dT/dz|   [W/mK]
    """

    def __init__(self, atoms, calc, n_slabs=20, swap_interval=20, temperature=300):
        self.atoms         = atoms
        self.calc          = calc
        self.n_slabs       = n_slabs
        self.swap_interval = swap_interval
        self.temperature   = temperature

        self.atoms.calc = calc

        # Cell geometry
        cell = np.array(atoms.get_cell())
        self.Lz = cell[2, 2]           # length along z (assumes orthorhombic)
        self.A  = cell[0, 0] * cell[1, 1]  # cross-sectional area (Å²)

        self.total_ke_transferred = 0.0   # eV
        self.n_swaps = 0
        self.step    = 0

        # Slab temperature accumulators (activated after equilibration)
        self.T_slab_sum  = np.zeros(n_slabs)
        self.T_slab_n    = np.zeros(n_slabs, dtype=int)
        self.kappa_log   = []           # (step, kappa_estimate)

    # ── slab assignment ─────────────────────────────────────────────────────

    def _get_slab_ids(self):
        z = self.atoms.positions[:, 2] % self.Lz
        return (z / self.Lz * self.n_slabs).astype(int) % self.n_slabs

    # ── velocity exchange ────────────────────────────────────────────────────

    def _swap(self):
        """
        Exchange z-velocities between the fastest atom in slab 0 (cold)
        and the slowest atom in slab N//2 (hot).
        Returns kinetic energy transferred (eV) to the hot slab.
        """
        slab_ids = self._get_slab_ids()
        vel      = self.atoms.get_velocities()    # Å/fs
        masses   = self.atoms.get_masses()        # amu

        cold_idx = np.where(slab_ids == 0)[0]
        hot_idx  = np.where(slab_ids == self.n_slabs // 2)[0]

        if len(cold_idx) == 0 or len(hot_idx) == 0:
            return 0.0

        # KE in z-direction (ASE units: amu·Å²/fs²)
        ke_z_cold = 0.5 * masses[cold_idx] * vel[cold_idx, 2] ** 2
        ke_z_hot  = 0.5 * masses[hot_idx]  * vel[hot_idx,  2] ** 2

        i = cold_idx[np.argmax(ke_z_cold)]   # fastest in cold
        j = hot_idx [np.argmin(ke_z_hot)]    # slowest in hot

        # Swap z-velocity components
        vz_i, vz_j = vel[i, 2], vel[j, 2]
        vel[i, 2], vel[j, 2] = vz_j, vz_i
        self.atoms.set_velocities(vel)

        # Energy transferred to hot slab (gained by atom j)
        dKE_amu = 0.5 * masses[j] * (vz_i ** 2 - vz_j ** 2)
        dKE_eV  = dKE_amu * units.fs ** 2 / units._amu / units._e * 1e-20  # amu·Å²/fs² → eV
        # Simpler: use ASE unit conversion
        dKE_eV  = dKE_amu * (units.fs ** 2) * units._amu / (units._e * 1e-20)

        # Most reliable: directly use ASE energy units
        # amu * (Å/fs)^2 = amu * 1e-20 m²/s² → to eV: ×1e-20 * 1.6606e-27 / 1.602e-19
        _amu_to_eV = 1.0364e-4   # amu·(Å/fs)² → eV
        dKE_eV = dKE_amu * _amu_to_eV

        self.total_ke_transferred += max(0.0, dKE_eV)
        self.n_swaps += 1
        return max(0.0, dKE_eV)

    # ── slab temperature ─────────────────────────────────────────────────────

    def _measure_slab_temperatures(self):
        """Compute instantaneous temperature in each slab."""
        slab_ids = self._get_slab_ids()
        vel      = self.atoms.get_velocities()
        masses   = self.atoms.get_masses()
        kB       = units.kB

        T_slab = np.zeros(self.n_slabs)
        for s in range(self.n_slabs):
            idx = np.where(slab_ids == s)[0]
            if len(idx) < 2:
                T_slab[s] = np.nan
                continue
            ke = 0.5 * np.sum(masses[idx, None] * vel[idx] ** 2) * 1.0364e-4  # eV
            T_slab[s] = 2 * ke / (3 * len(idx) * kB)

        return T_slab

    # ── observer ─────────────────────────────────────────────────────────────

    def make_observer(self, equil_steps, record_interval=100):
        """Return a callable observer to attach to the ASE MD object."""
        def observer():
            self.step += 1

            # Velocity swap
            if self.step % self.swap_interval == 0:
                self._swap()

            # Accumulate slab temperatures after equilibration
            if self.step > equil_steps and self.step % record_interval == 0:
                T_slab = self._measure_slab_temperatures()
                valid  = ~np.isnan(T_slab)
                self.T_slab_sum[valid] += T_slab[valid]
                self.T_slab_n[valid]   += 1

                # Running κ estimate
                if self.total_ke_transferred > 0 and self.T_slab_n.min() > 2:
                    kappa = self._estimate_kappa(self.step * 1e-3)  # rough ps estimate
                    if kappa is not None:
                        self.kappa_log.append((self.step, kappa))

        return observer

    # ── thermal conductivity ─────────────────────────────────────────────────

    def _estimate_kappa(self, elapsed_ps):
        """
        Estimate κ from current accumulated T_slab and total transferred energy.
        Returns κ in W/mK, or None if insufficient data.
        """
        mask = self.T_slab_n > 0
        if mask.sum() < 4:
            return None

        T_mean = np.where(mask, self.T_slab_sum / np.maximum(self.T_slab_n, 1), np.nan)
        z_centers = (np.arange(self.n_slabs) + 0.5) / self.n_slabs * self.Lz

        valid = ~np.isnan(T_mean)
        if valid.sum() < 4:
            return None

        # Fit dT/dz to two halves separately (periodic: hot at L/2, cold at 0 and L)
        half = self.n_slabs // 2
        # Lower half: z ∈ [0, L/2] — gradient should be positive (cold→hot)
        z_lo, T_lo = z_centers[:half][valid[:half]], T_mean[:half][valid[:half]]
        # Upper half: z ∈ [L/2, L] — gradient should be negative (hot→cold)
        z_hi, T_hi = z_centers[half:][valid[half:]], T_mean[half:][valid[half:]]

        if len(z_lo) < 2 or len(z_hi) < 2:
            return None

        grad_lo = np.polyfit(z_lo, T_lo, 1)[0]   # K/Å
        grad_hi = np.polyfit(z_hi, T_hi, 1)[0]   # K/Å
        dT_dz   = 0.5 * (abs(grad_lo) + abs(grad_hi))   # K/Å

        if dT_dz < 1e-6:
            return None

        # Heat flux J (eV/Å²/ps) — factor 2 for bidirectional flow
        t_total_ps = elapsed_ps
        J_eV = self.total_ke_transferred / (2 * t_total_ps * self.A)  # eV/Å²/ps

        # Convert: eV/Å²/ps → W/m²  (1 eV/Å²/ps = 1.602e-19 / (1e-10)^2 / 1e-12 W/m²)
        J_Wm2 = J_eV * 1.602e-19 / (1e-10 ** 2) / 1e-12   # W/m²
        grad_Km = dT_dz * 1e10                              # K/Å → K/m

        kappa = J_Wm2 / grad_Km                             # W/mK
        return kappa

    def finalize(self, total_steps, timestep_fs):
        """Compute final κ, write outputs."""
        elapsed_ps = total_steps * timestep_fs * 1e-3

        mask   = self.T_slab_n > 0
        T_mean = np.where(mask, self.T_slab_sum / np.maximum(self.T_slab_n, 1), np.nan)
        z_centers = (np.arange(self.n_slabs) + 0.5) / self.n_slabs * self.Lz

        # Write temperature profile
        with open("temperature_profile.dat", "w") as f:
            f.write(f"# Müller-Plathe rNEMD temperature profile\n")
            f.write(f"# T_target={self.temperature} K, n_slabs={self.n_slabs}\n")
            f.write(f"{'#slab':>6}  {'z(A)':>10}  {'T(K)':>10}  {'n_samples':>10}\n")
            for s, (z, T, n) in enumerate(zip(z_centers, T_mean, self.T_slab_n)):
                f.write(f"{s:6d}  {z:10.4f}  {T:10.4f}  {n:10d}\n")
        print("Wrote temperature_profile.dat")

        # Write convergence log
        if self.kappa_log:
            with open("kappa_convergence.dat", "w") as f:
                f.write(f"{'#step':>10}  {'kappa(W/mK)':>14}\n")
                for step, k in self.kappa_log:
                    f.write(f"{step:10d}  {k:14.6f}\n")
            print("Wrote kappa_convergence.dat")

        # Final κ
        kappa = self._estimate_kappa(elapsed_ps)

        # Plot temperature profile
        valid = ~np.isnan(T_mean)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(z_centers[valid], T_mean[valid], "o-", ms=5)
        ax.axvline(self.Lz / 2, ls="--", c="r", alpha=0.5, label="hot slab")
        ax.axvline(0,           ls="--", c="b", alpha=0.5, label="cold slab")
        ax.set_xlabel("z (Å)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(f"rNEMD temperature profile  (T_target={self.temperature} K)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("temperature_profile.png", dpi=300)
        plt.close(fig)
        print("Wrote temperature_profile.png")

        # Summary
        with open("kappa_summary.txt", "w") as f:
            f.write("Müller-Plathe rNEMD — thermal conductivity summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Target temperature    : {self.temperature} K\n")
            f.write(f"Total steps           : {total_steps}\n")
            f.write(f"Timestep              : {timestep_fs} fs\n")
            f.write(f"Total time            : {elapsed_ps:.3f} ps\n")
            f.write(f"Swaps performed       : {self.n_swaps}\n")
            f.write(f"Total ΔKE transferred : {self.total_ke_transferred:.6f} eV\n")
            f.write(f"κ (W/mK)              : {kappa:.4f}\n" if kappa else "κ: insufficient data\n")
        print("Wrote kappa_summary.txt")
        if kappa:
            print(f"\nThermal conductivity κ = {kappa:.4f} W/mK")
        return kappa


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lattice thermal conductivity via Müller-Plathe reverse NEMD with GRACE."
    )
    parser.add_argument("--poscar",       default="POSCAR",       help="Input structure.")
    parser.add_argument("--model",        default="GRACE-2L-OAM", help="GRACE model.")
    parser.add_argument("--temperature",  type=float, default=300, help="Target T in K.")
    parser.add_argument("--steps",        type=int, default=200000,help="Total MD steps.")
    parser.add_argument("--equil_frac",   type=float, default=0.4, help="Fraction of steps for equilibration (default 0.4).")
    parser.add_argument("--timestep",     type=float, default=1.0, help="MD timestep in fs (default 1.0).")
    parser.add_argument("--nslabs",       type=int, default=20,    help="Number of slabs (default 20).")
    parser.add_argument("--swap_interval",type=int, default=20,    help="Steps between swaps (default 20).")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    atoms = read(args.poscar, format="vasp")
    print(f"Loaded: {len(atoms)} atoms from {args.poscar}")

    # Load GRACE model
    if os.path.exists(args.model):
        from tensorpotential.calculator import TPCalculator
        calc = TPCalculator(args.model)
    else:
        from tensorpotential.calculator.foundation_models import grace_fm
        calc = grace_fm(args.model)

    # Initialise velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    Stationary(atoms)
    ZeroRotation(atoms)

    equil_steps = int(args.steps * args.equil_frac)
    prod_steps  = args.steps - equil_steps

    print(f"\n--- Müller-Plathe rNEMD ---")
    print(f"Temperature   : {args.temperature} K")
    print(f"Total steps   : {args.steps}  (equil={equil_steps}, prod={prod_steps})")
    print(f"Timestep      : {args.timestep} fs")
    print(f"Slabs         : {args.nslabs}")
    print(f"Swap interval : {args.swap_interval} steps\n")

    nemd = MuellerPlatheNEMD(
        atoms, calc,
        n_slabs=args.nslabs,
        swap_interval=args.swap_interval,
        temperature=args.temperature,
    )

    dt = args.timestep * units.fs
    md = VelocityVerlet(atoms, timestep=dt)
    observer = nemd.make_observer(equil_steps, record_interval=50)
    md.attach(observer, interval=1)

    print("Running MD...")
    md.run(args.steps)

    kappa = nemd.finalize(args.steps, args.timestep)
    write("CONTCAR_kappa", atoms, format="vasp")
    print("Final structure written to CONTCAR_kappa")


if __name__ == "__main__":
    main()
