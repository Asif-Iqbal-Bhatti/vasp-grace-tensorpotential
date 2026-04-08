import os
import numpy as np
from ase.io import write

def format_outcar_block(atoms, energy, forces, stress_voigt, step):
    """Formats a single iteration block for the OUTCAR file."""
    eV_to_kB = 1602.1766208
    
    # ASE returns Voigt stress in order: xx, yy, zz, yz, xz, xy (in eV/A^3)
    # VASP OUTCAR writes stress in order: XX, YY, ZZ, XY, YZ, ZX (in kB)
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
    """Safely fetch stress, handling potentials that might not implement it."""
    try:
        stress = atoms.get_stress(voigt=True)
        return stress if stress is not None else np.zeros(6)
    except Exception:
        return np.zeros(6)


class VaspWriterObserver:
    """An ASE Optimizer observer to mimic VASP's step-by-step file generation."""
    def __init__(self, atoms):
        self.atoms = atoms
        self.step = 1
        open("OSZICAR", "w").close()
        open("OUTCAR", "w").close()
        
    def __call__(self):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        stress = safe_get_stress(self.atoms)
        
        # Append iteration to OSZICAR
        with open("OSZICAR", "a") as f:
            f.write(f"   {self.step:4d} F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n")
            
        # Append iteration to OUTCAR
        with open("OUTCAR", "a") as f:
            f.write(format_outcar_block(self.atoms, energy, forces, stress, self.step))
            
        # Update CONTCAR continuously
        write("CONTCAR", self.atoms, format="vasp")
        self.step += 1


def write_vasp_single_point(atoms, energy, forces, stress_voigt):
    """Outputs VASP files for a single point calculation."""
    with open("OSZICAR", "w") as f:
        f.write(f"   1 F= {energy:15.8E} E0= {energy:15.8E}  d E ={0.0:15.8E}\n")
        
    with open("OUTCAR", "w") as f:
        f.write(format_outcar_block(atoms, energy, forces, stress_voigt, 1))
        
    write("CONTCAR", atoms, format="vasp")
