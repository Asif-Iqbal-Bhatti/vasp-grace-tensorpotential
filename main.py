#!/usr/bin/env python3
"""
vasp_grace.py

A standalone Python script to use GRACE Machine Learning Potentials 
as a drop-in replacement for VASP. It supports Single Point calculations, 
Geometry Optimizations, and Molecular Dynamics natively using ASE.
"""

import os
import sys
import argparse
import numpy as np

from ase.io import read, write
from ase.optimize import LBFGS, FIRE2
from ase.filters import FrechetCellFilter
from ase import units

# Molecular Dynamics Imports
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.andersen import Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nose_hoover_chain import MTKNPT

# GRACE / TensorPotential Imports
try:
    from tensorpotential.calculator.foundation_models import grace_fm
    from tensorpotential.calculator import TPCalculator
except ImportError:
    print("Error: 'tensorpotential' is not installed. Install via: pip install tensorpotential")
    sys.exit(1)


# ==========================================
# 1. Output Formatting (Mock VASP Files)
# ==========================================

def format_outcar_block(atoms, energy, forces, stress_voigt, step):
    """Formats a single iteration block for the OUTCAR file mimicking VASP format."""
    eV_to_kB = 1602.1766208
    s_xx, s_yy, s_zz, s_yz, s_xz, s_xy = stress_voigt * eV_to_kB
    
    lines =[
        "--------------------------------------------------------------------------------",
        f" STEP {step}",
        "--------------------------------------------------------------------------------",
        "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        "  ---------------------------------------------------",
        f"  free  energy   TOTEN  = {energy:18.8f} eV\n",
        " POSITION                                       TOTAL-FORCE (eV/Angst)",
        " -----------------------------------------------------------------------------------"
    ]
    
    for pos, force in zip(atoms.positions, forces):
        lines.append(f" {pos[0]:12.5f} {pos[1]:12.5f} {pos[2]:12.5f}    {force[0]:12.5f} {force[1]:12.5f} {force[2]:12.5f}")
    
    lines.append(" -----------------------------------------------------------------------------------\n")
    lines.append("  FORCE on cell =-STRESS in cart. coord.  units (eV):")
    lines.append("  Direction    XX          YY          ZZ          XY          YZ          ZX")
    lines.append("  --------------------------------------------------------------------------------------")
    lines.append(f"  in kB     {s_xx:10.5f}  {s_yy:10.5f}  {s_zz:10.5f}  {s_xy:10.5f}  {s_yz:10.5f}  {s_xz:10.5f}\n")
    
    return "\n".join(lines) + "\n"

def safe_get_stress(atoms):
    """Safely fetch stress, returning zeros if the cell is non-periodic or undefined."""
    try:
        stress = atoms.get_stress(voigt=True)
        return stress if stress is not None else np.zeros(6)
    except Exception:
        return np.zeros(6)

class VaspWriterObserver:
    """Observer attached to ASE Optimizers/MD to write VASP output files iteratively."""
    def __init__(self, atoms, is_md=False):
        self.atoms = atoms
        self.step = 1
        self.is_md = is_md
        open("OSZICAR", "w").close()
        open("OUTCAR", "w").close()
        
    def __call__(self):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        stress = safe_get_stress(self.atoms)
        
        with open("OSZICAR", "a") as f:
            if self.is_md:
                temp = self.atoms.get_temperature()
                kin_e = self.atoms.get_kinetic_energy()
                tot_e = energy + kin_e
                f.write(f"   {self.step:4d} T= {temp:8.2f} E= {tot_e:15.8E} F= {energy:15.8E} EK= {kin_e:15.8E}\n")
            else:
                f.write(f"   {self.step:4d} F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n")
            
        with open("OUTCAR", "a") as f:
            f.write(format_outcar_block(self.atoms, energy, forces, stress, self.step))
            
        write("CONTCAR", self.atoms, format="vasp")
        self.step += 1

def write_vasp_single_point(atoms, energy, forces, stress_voigt):
    """Handles static output files for single-point calculations."""
    with open("OSZICAR", "w") as f:
        f.write(f"   1 F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n")
        
    with open("OUTCAR", "w") as f:
        f.write(format_outcar_block(atoms, energy, forces, stress_voigt, 1))
        
    write("CONTCAR", atoms, format="vasp")

def generate_dummy_potcar(atoms):
    """Generates a dummy POTCAR to prevent downstream VASP parsers from failing."""
    if not os.path.exists("POTCAR"):
        symbols = set(atoms.get_chemical_symbols())
        with open("POTCAR", "w") as f:
            for elem in symbols:
                f.write(f" PAW_PBE {elem} \n")


# ==========================================
# 2. Input Parsing (INCAR) & ML Setup
# ==========================================

def parse_incar(filepath="INCAR"):
    """Reads VASP INCAR parameters and custom GRACE flags."""
    params = {
        "IBRION": -1,
        "NSW": 0,
        "ISIF": 2,
        "EDIFFG": -0.01,
        "GRACE_MODEL": "GRACE-1L-OMAT-medium-ft-E", # Default GRACE model
        "POTIM": 1.0,
        "TEBEG": 300.0,
        "MDALGO": 0
    }
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.split("#")[0].split("!")[0].strip()
                if "=" in line:
                    key, val = [x.strip() for x in line.split("=", 1)]
                    key = key.upper()
                    if key in params:
                        # Auto-cast logic based on default types
                        if isinstance(params[key], int):
                            params[key] = int(val)
                        elif isinstance(params[key], float):
                            params[key] = float(val)
                        else:
                            params[key] = val
    return params

def get_calculator(model_identifier):
    """Loads the appropriate GRACE ML potential."""
    if os.path.exists(model_identifier):
        print(f"Loading custom GRACE model from path: {model_identifier}")
        return TPCalculator(model_identifier)
    else:
        print(f"Loading GRACE foundation model: {model_identifier}")
        return grace_fm(model_identifier)


# ==========================================
# 3. Main Execution Routine
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Mock VASP execution using GRACE Machine Learning Potential.")
    parser.add_argument("--poscar", default="POSCAR", help="Path to POSCAR file.")
    parser.add_argument("--incar", default="INCAR", help="Path to INCAR file.")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found! Run this script from a VASP working directory.")
        sys.exit(1)

    print("--- vasp-grace initialized ---")

    atoms = read(args.poscar, format="vasp")
    incar = parse_incar(args.incar)
    generate_dummy_potcar(atoms)
    
    # Attach GRACE Calculator
    calc = get_calculator(incar["GRACE_MODEL"])
    atoms.calc = calc
    
    # Extract run controls
    ibrion = incar["IBRION"]
    nsw    = incar["NSW"]
    isif   = incar["ISIF"]
    ediffg = incar["EDIFFG"]
    fmax   = abs(ediffg) if ediffg < 0 else 0.05
    
    # --- ROUTINE A: Single Point ---
    if nsw <= 0 or ibrion == -1:
        print("Performing Single-Point Evaluation...")
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = safe_get_stress(atoms)
        write_vasp_single_point(atoms, energy, forces, stress)
        print("Done. Wrote OUTCAR, OSZICAR, CONTCAR.")
        
    # --- ROUTINE B: Molecular Dynamics ---
    elif ibrion == 0:
        print(f"Performing Molecular Dynamics (IBRION=0, TEBEG={incar['TEBEG']}K, POTIM={incar['POTIM']}fs)...")
        
        # Initialize velocities randomly
        if incar["TEBEG"] > 0:
            MaxwellBoltzmannDistribution(atoms, temperature_K=incar["TEBEG"])
            Stationary(atoms)
            ZeroRotation(atoms)

        dt = incar["POTIM"] * units.fs
        temperature = incar["TEBEG"]

        if incar["ISIF"] >= 3:
            print("Using NPT Ensemble (ISIF=3)...")
            target_pressure = 1.0 * units.bar 
            md = MTKNPT(atoms, timestep=dt, temperature_K=temperature, 
                     pressure_au=target_pressure, tdamp = 100*dt, pdamp = 1000*dt)
        else:
            if incar["MDALGO"] == 0:
                print("Using NVE Ensemble (Velocity Verlet, MDALGO=0)...")
                md = VelocityVerlet(atoms, timestep=dt)
            elif incar["MDALGO"] == 1:
                print("Using NVT Ensemble (Andersen Thermostat, MDALGO=1)...")
                md = Andersen(atoms, timestep=dt, temperature_K=temperature, andersen_prob=1e-4)
            elif incar["MDALGO"] == 2:
                print("Using NVT Ensemble (Berendsen Thermostat, MDALGO=2)...")
                md = NVTBerendsen(atoms, timestep=dt, temperature_K=temperature, taut=100*units.fs)
            else:
                print("Fallback: Using NVE Ensemble (Velocity Verlet)...")
                md = VelocityVerlet(atoms, timestep=dt)

        observer = VaspWriterObserver(atoms, is_md=True)
        md.attach(observer, interval=1)
        observer() # Write step 0
        md.run(nsw)
        print("MD successfully completed.")

    # --- ROUTINE C: Geometry Optimization ---
    else:
        print(f"Performing Geometry Optimization (IBRION={ibrion}, NSW={nsw}, ISIF={isif})...")
        
        if isif == 3:
            opt_target = FrechetCellFilter(atoms)
        elif isif == 4:
            opt_target = FrechetCellFilter(atoms, constant_volume=True)
        else:
            opt_target = atoms
            
        if ibrion == 1:
            optimizer = LBFGS(opt_target, trajectory='grace_opt.traj', logfile='grace_opt.log')
        else:
            optimizer = FIRE2(opt_target, trajectory='grace_opt.traj', logfile='grace_opt.log')
            
        observer = VaspWriterObserver(atoms, is_md=False)
        optimizer.attach(observer, interval=1)
        observer() # Write step 0
        optimizer.run(fmax=fmax, steps=nsw)
        print("Optimization successfully completed.")

if __name__ == "__main__":
    main()
