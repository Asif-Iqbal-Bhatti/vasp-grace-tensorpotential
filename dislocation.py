#!/usr/bin/env python3
"""
dislocation.py

Build edge, screw, and mixed dislocation structures for GRACE MLIP calculations.

Applies the Volterra displacement field (isotropic elasticity) or the full Stroh
anisotropic formalism (when elastic constants are provided) to create dislocation
structures from a perfect crystal POSCAR — similar in spirit to atomsk's
--dislocation mode and atomman's dislocation module.

Coordinate convention
---------------------
    ξ  = dislocation line direction  → aligned with z-axis of the cell
    n  = slip plane normal           → aligned with y-axis
    m  = n × ξ  (in-plane, ⊥ line)  → aligned with x-axis

    Edge  : b = b_magnitude * x̂  (Burgers ⊥ line, in slip plane)
    Screw : b = b_magnitude * ẑ  (Burgers ∥ line)
    Mixed : b = b_x * x̂ + b_z * ẑ

Usage
-----
    # Screw dislocation dipole, isotropic
    python dislocation.py --poscar POSCAR --type screw --burgers 2.8

    # Edge dislocation, specify Poisson ratio
    python dislocation.py --poscar POSCAR --type edge --burgers 2.8 --poisson 0.28

    # Anisotropic using elastic constants from the elastic module
    python dislocation.py --poscar POSCAR --type edge --burgers 2.8 \\
                          --elastic ELASTIC_Cij_GPa.dat

    # Specify slip plane normal and slip direction as Cartesian unit vectors
    python dislocation.py --poscar POSCAR --type screw --burgers 2.8 \\
                          --line 0 0 1 --normal 0 1 0

    # Miller index input (plane normal and line direction)
    python dislocation.py --poscar POSCAR --type edge --burgers 2.8 \\
                          --hkl 1 1 0 --uvw 1 -1 0

    # Build then relax with GRACE
    python dislocation.py --poscar POSCAR --type screw --burgers 2.8 \\
                          --relax --model GRACE-2L-OAM

Outputs
-------
    POSCAR_dislocation   : structure with Volterra displacement applied
    dislocation_info.txt : dislocation parameters and energy (if relaxed)
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
from ase import units


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Elastic tensor utilities
# ──────────────────────────────────────────────────────────────────────────────

# Voigt index mapping: (i,j) → Voigt index (0-based)
_VOIGT_MAP = {
    (0, 0): 0, (1, 1): 1, (2, 2): 2,
    (1, 2): 3, (2, 1): 3,
    (0, 2): 4, (2, 0): 4,
    (0, 1): 5, (1, 0): 5,
}


def voigt_to_full(C_voigt):
    """Convert 6×6 Voigt elastic tensor to 3×3×3×3 full tensor."""
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = C_voigt[_VOIGT_MAP[(i, j)], _VOIGT_MAP[(k, l)]]
    return C


def rotate_elastic_tensor(C_full, R):
    """
    Rotate 3×3×3×3 elastic tensor by rotation matrix R.
    C'_{ijkl} = R_{ia} R_{jb} R_{kc} R_{ld} C_{abcd}
    """
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, C_full)


def extract_QRT(C_full):
    """
    Extract Q, R, T Stroh sub-matrices from full elastic tensor.
    Assumes dislocation frame: x=0, y=1, z=2 (line direction).
    Q_{ik} = C_{i,x,k,x}  (= C_{i,0,k,0})
    R_{ik} = C_{i,x,k,y}  (= C_{i,0,k,1})
    T_{ik} = C_{i,y,k,y}  (= C_{i,1,k,1})
    """
    Q = C_full[:, 0, :, 0].copy()
    R = C_full[:, 0, :, 1].copy()
    T = C_full[:, 1, :, 1].copy()
    return Q, R, T


def parse_elastic_file(filename):
    """
    Parse elastic constants from ELASTIC_Cij_GPa.dat (produced by main.py)
    or a plain 6×6 whitespace-separated matrix file.

    Returns a 6×6 numpy array in GPa.
    """
    with open(filename) as f:
        lines = f.readlines()

    # Detect format: labelled (C11 = value) vs raw matrix
    labelled = any("=" in l for l in lines)

    if labelled:
        C_voigt = np.zeros((6, 6))
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "===" in line or line.startswith("Elastic"):
                continue
            if "=" in line:
                name, val = line.split("=")
                name = name.strip()          # e.g. "C11"
                val = float(val.strip().split()[0])
                # Parse Cij → 0-based Voigt indices
                digits = [int(c) - 1 for c in name[1:]]
                if len(digits) == 2:
                    i, j = digits
                    C_voigt[i, j] = val
                    C_voigt[j, i] = val
        return C_voigt
    else:
        # Try reading as plain 6×6 matrix (skip comment lines)
        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = [float(x) for x in line.split()]
            if len(row) == 6:
                data.append(row)
        if len(data) == 6:
            return np.array(data)
        raise ValueError(
            f"Could not parse elastic constant file: {filename}\n"
            "Expected either ELASTIC_Cij_GPa.dat format or a plain 6×6 matrix."
        )


def miller_to_cartesian(indices, cell, reciprocal=False):
    """
    Convert Miller indices to a Cartesian vector using the unit cell.

    Direction [uvw] → u*a + v*b + w*c                  (reciprocal=False)
    Plane normal (hkl) → h*a* + k*b* + l*c*            (reciprocal=True)
    """
    cell = np.array(cell)
    indices = np.array(indices, dtype=float)
    if not reciprocal:
        vec = indices @ cell
    else:
        # Reciprocal lattice vectors
        vol = np.dot(cell[0], np.cross(cell[1], cell[2]))
        rec = np.array([
            np.cross(cell[1], cell[2]) / vol,
            np.cross(cell[2], cell[0]) / vol,
            np.cross(cell[0], cell[1]) / vol,
        ])
        vec = indices @ rec
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError(f"Miller index {indices} maps to a zero vector.")
    return vec / norm


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Isotropic Volterra displacement field
# ──────────────────────────────────────────────────────────────────────────────

class IsotropicVolterra:
    """
    Volterra displacement field for edge, screw, and mixed dislocations
    in an isotropic elastic medium.

    Coordinate frame:
        x (axis 0) : m direction  (⊥ line, in slip plane)
        y (axis 1) : n direction  (slip plane normal)
        z (axis 2) : ξ direction  (dislocation line)

    All positions and displacements are in the same units as `burgers`.
    """

    def __init__(self, poisson_ratio=0.3):
        self.nu = poisson_ratio

    def screw(self, xy, center=(0.0, 0.0)):
        """
        Screw component: b along z.
        u_z = (b/2π) arctan2(y - y0, x - x0)
        Returns displacement vector [0, 0, u_z] (normalized by b).
        """
        x = xy[:, 0] - center[0]
        y = xy[:, 1] - center[1]
        u_z = np.arctan2(y, x) / (2.0 * np.pi)
        disp = np.zeros((len(xy), 3))
        disp[:, 2] = u_z
        return disp

    def edge(self, xy, center=(0.0, 0.0)):
        """
        Edge component: b along x.
        u_x = (b/2π) [ arctan2(y,x) + xy / (2(1-ν)(x²+y²)) ]
        u_y = -(b/2π) [ (1-2ν)/(4(1-ν)) ln(x²+y²) + (x²-y²)/(4(1-ν)(x²+y²)) ]
        Returns displacement [u_x, u_y, 0] (normalized by b).
        Atoms at the core (r < 0.1 Å) are not displaced to avoid singularity.
        """
        nu = self.nu
        x = xy[:, 0] - center[0]
        y = xy[:, 1] - center[1]
        r2 = x**2 + y**2
        safe = r2 > 1e-4

        u_x = np.zeros(len(xy))
        u_y = np.zeros(len(xy))

        theta = np.arctan2(y[safe], x[safe])
        rs2 = r2[safe]

        u_x[safe] = (theta + x[safe] * y[safe] / (2.0 * (1.0 - nu) * rs2)) / (2.0 * np.pi)
        u_y[safe] = -(
            (1.0 - 2.0 * nu) / (4.0 * (1.0 - nu)) * np.log(rs2)
            + (x[safe]**2 - y[safe]**2) / (4.0 * (1.0 - nu) * rs2)
        ) / (2.0 * np.pi)

        disp = np.zeros((len(xy), 3))
        disp[:, 0] = u_x
        disp[:, 1] = u_y
        return disp

    def compute(self, xy, disloc_type, center=(0.0, 0.0)):
        """Return displacement field (normalized by b) for a given type."""
        if disloc_type == "screw":
            return self.screw(xy, center)
        elif disloc_type == "edge":
            return self.edge(xy, center)
        elif disloc_type == "mixed":
            return self.screw(xy, center) + self.edge(xy, center)
        else:
            raise ValueError(f"Unknown dislocation type: {disloc_type}")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Stroh anisotropic formalism
# ──────────────────────────────────────────────────────────────────────────────

class StrohAnisotropic:
    """
    Anisotropic Stroh displacement field for a straight dislocation.

    Implements the Stroh (1958) / Barnett-Lothe formalism for arbitrary
    crystal symmetry. Reference: Hirth & Lothe, Theory of Dislocations, Ch.13;
    Ting (1996), Anisotropic Elasticity.

    The elastic tensor C_voigt (6×6, GPa) is rotated to the dislocation
    coordinate frame (m, n, ξ) before solving the eigenvalue problem.
    """

    def __init__(self, C_voigt_GPa, rotation=None):
        """
        Parameters
        ----------
        C_voigt_GPa : (6,6) array
            Elastic stiffness in Voigt notation (GPa).
        rotation : (3,3) array or None
            Rotation matrix from crystal frame to dislocation frame
            (rows = [m̂, n̂, ξ̂]).  Identity if None.
        """
        C_full = voigt_to_full(C_voigt_GPa)
        if rotation is not None:
            R = np.array(rotation)
            C_full = rotate_elastic_tensor(C_full, R.T)  # R^T rotates crystal→dislocframe

        Q, R, T = extract_QRT(C_full)
        self._setup_stroh(Q, R, T)

    def _setup_stroh(self, Q, R, T):
        """Solve the 6×6 Stroh eigenvalue problem and store results."""
        T_inv = np.linalg.inv(T)

        # Stroh N matrix (Ting 1996 convention)
        N = np.block([
            [-T_inv @ R.T,          T_inv],
            [R @ T_inv @ R.T - Q,  -R @ T_inv],
        ])

        eigvals, eigvecs = np.linalg.eig(N)

        # Keep the 3 eigenvalues with positive imaginary part
        pos_idx = np.where(eigvals.imag > 1e-10)[0]
        if len(pos_idx) < 3:
            # Fall back to largest imaginary parts
            pos_idx = np.argsort(eigvals.imag)[-3:]

        self.p = eigvals[pos_idx]               # (3,) complex eigenvalues
        xi = eigvecs[:, pos_idx]                 # (6,3) eigenvectors
        self.A = xi[:3, :]                       # (3,3) displacement amplitudes
        self.L = xi[3:, :]                       # (3,3) traction amplitudes

    def _solve_q(self, burgers):
        """
        Find the complex amplitudes q from the Burgers vector condition:
            2 Re[A q] = b
            2 Re[L q] = 0  (no net traction)
        Solved as a real 6×6 linear system.
        """
        A_r, A_i = self.A.real, self.A.imag
        L_r, L_i = self.L.real, self.L.imag

        M = np.block([
            [A_r, -A_i],
            [L_r, -L_i],
        ])
        rhs = np.concatenate([burgers / 2.0, np.zeros(3)])

        try:
            ql = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            ql = np.linalg.lstsq(M, rhs, rcond=None)[0]

        q = ql[:3] + 1j * ql[3:]
        return q

    def displacement(self, xy, burgers, center=(0.0, 0.0)):
        """
        Compute displacement field at positions xy (N×2 array).

        Parameters
        ----------
        xy      : (N,2) array of (x, y) positions in dislocation frame.
        burgers : (3,) Burgers vector in dislocation frame (same units as xy).
        center  : (x0, y0) position of dislocation core.

        Returns (N,3) displacement array.
        """
        q = self._solve_q(np.asarray(burgers, dtype=float))

        x = xy[:, 0] - center[0]
        y = xy[:, 1] - center[1]

        u = np.zeros((len(xy), 3))
        for alpha in range(3):
            z = x + self.p[alpha] * y          # complex z_α = x + p_α y
            f = np.log(z) / (2j * np.pi)       # complex log, branch cut at Im=0, Re<0
            contrib = q[alpha] * np.outer(f, self.A[:, alpha])  # (N,3) complex
            u += 2.0 * contrib.real

        return u


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Dislocation builder
# ──────────────────────────────────────────────────────────────────────────────

class DislocationBuilder:
    """
    Apply Volterra or Stroh displacement to a perfect crystal supercell.

    The dislocation line is assumed to run along the z-axis of the cell.
    A dipole configuration (two dislocations of opposite sign at ±Lx/4)
    is used by default to maintain 3D periodicity.
    """

    def __init__(self, atoms, method="isotropic", poisson_ratio=0.3,
                 elastic_file=None, line_dir=None, plane_normal=None):
        """
        Parameters
        ----------
        atoms         : ASE Atoms (perfect crystal supercell).
        method        : 'isotropic' or 'anisotropic'.
        poisson_ratio : ν for isotropic Volterra.
        elastic_file  : path to ELASTIC_Cij_GPa.dat for anisotropic Stroh.
        line_dir      : (3,) Cartesian unit vector for dislocation line (ξ̂).
                        Default: z-axis [0,0,1].
        plane_normal  : (3,) Cartesian unit vector for slip plane normal (n̂).
                        Default: y-axis [0,1,0].
        """
        self.atoms = atoms.copy()
        self.method = method
        self.nu = poisson_ratio

        # Dislocation frame axes
        cell = np.array(atoms.get_cell())
        self.xi = self._normalise(line_dir if line_dir is not None else [0, 0, 1])
        self.n  = self._normalise(plane_normal if plane_normal is not None else [0, 1, 0])
        self.m  = self._normalise(np.cross(self.n, self.xi))   # in-plane ⊥ line

        # Rotation matrix: rows = [m̂, n̂, ξ̂]
        self.R_frame = np.array([self.m, self.n, self.xi])

        if method == "anisotropic":
            if elastic_file is None or not os.path.exists(elastic_file):
                raise FileNotFoundError(
                    f"Anisotropic Stroh requires an elastic constant file. "
                    f"Provide --elastic path or run the elastic module first."
                )
            C_voigt = parse_elastic_file(elastic_file)
            self.stroh = StrohAnisotropic(C_voigt, rotation=self.R_frame)
        else:
            self.volterra = IsotropicVolterra(poisson_ratio)

    @staticmethod
    def _normalise(v):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < 1e-12:
            raise ValueError(f"Zero vector provided: {v}")
        return v / n

    def _get_xy_positions(self):
        """Project atom positions onto (m̂, n̂) plane → (N,2) array."""
        pos = self.atoms.positions
        x = pos @ self.m
        y = pos @ self.n
        return np.column_stack([x, y])

    def _apply_displacements(self, disp_nm, burgers_magnitude):
        """
        Convert normalised displacements (in units of b) back to Å
        and add to atom positions.
        """
        full_disp = np.zeros_like(self.atoms.positions)
        full_disp += burgers_magnitude * disp_nm[:, 0:1] * self.m
        full_disp += burgers_magnitude * disp_nm[:, 1:2] * self.n
        full_disp += burgers_magnitude * disp_nm[:, 2:3] * self.xi
        self.atoms.positions += full_disp

    def build_dipole(self, disloc_type, burgers_magnitude):
        """
        Create a dislocation dipole: +b at (Lm/4, Ln/2) and -b at (3Lm/4, Ln/2).

        The two dislocations of opposite sign maintain full 3D periodicity.

        Parameters
        ----------
        disloc_type      : 'edge', 'screw', or 'mixed'
        burgers_magnitude: magnitude of the Burgers vector in Å

        Returns the modified ASE Atoms object.
        """
        cell = np.array(self.atoms.get_cell())
        Lm = np.linalg.norm(cell @ self.m)  # cell length along m
        Ln = np.linalg.norm(cell @ self.n)  # cell length along n

        # Core positions in m-n coordinates
        c1 = (Lm * 0.25, Ln * 0.50)
        c2 = (Lm * 0.75, Ln * 0.50)

        xy = self._get_xy_positions()

        if self.method == "isotropic":
            d1 =  self.volterra.compute(xy, disloc_type, center=c1)
            d2 = -self.volterra.compute(xy, disloc_type, center=c2)
            self._apply_displacements(d1 + d2, burgers_magnitude)

        else:  # anisotropic Stroh
            # Burgers vector in dislocation frame
            if disloc_type == "edge":
                b_frame = np.array([burgers_magnitude, 0.0, 0.0])
            elif disloc_type == "screw":
                b_frame = np.array([0.0, 0.0, burgers_magnitude])
            else:
                b_frame = np.array([burgers_magnitude, 0.0, burgers_magnitude]) / np.sqrt(2)

            d1 =  self.stroh.displacement(xy, b_frame,  center=c1)
            d2 =  self.stroh.displacement(xy, -b_frame, center=c2)
            # Convert from dislocation-frame components back to Cartesian
            disp_cart = np.zeros_like(self.atoms.positions)
            combined = d1 + d2
            disp_cart += combined[:, 0:1] * self.m
            disp_cart += combined[:, 1:2] * self.n
            disp_cart += combined[:, 2:3] * self.xi
            self.atoms.positions += disp_cart

        return self.atoms

    def build_single(self, disloc_type, burgers_magnitude):
        """
        Apply a single dislocation at the cell centre.
        Note: a single dislocation is not compatible with periodic boundary
        conditions; use fixed boundaries in n̂ direction or a cluster model.
        """
        cell = np.array(self.atoms.get_cell())
        Lm = np.linalg.norm(cell @ self.m)
        Ln = np.linalg.norm(cell @ self.n)
        center = (Lm * 0.5, Ln * 0.5)
        xy = self._get_xy_positions()

        print("Warning: single dislocation breaks periodicity along n̂.")
        print("         Consider using build_dipole() or a cluster cell.")

        if self.method == "isotropic":
            d = self.volterra.compute(xy, disloc_type, center=center)
            self._apply_displacements(d, burgers_magnitude)
        else:
            if disloc_type == "edge":
                b_frame = np.array([burgers_magnitude, 0.0, 0.0])
            elif disloc_type == "screw":
                b_frame = np.array([0.0, 0.0, burgers_magnitude])
            else:
                b_frame = np.array([burgers_magnitude, 0.0, burgers_magnitude]) / np.sqrt(2)

            d = self.stroh.displacement(xy, b_frame, center=center)
            disp_cart = (d[:, 0:1] * self.m + d[:, 1:2] * self.n + d[:, 2:3] * self.xi)
            self.atoms.positions += disp_cart

        return self.atoms


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Output helper
# ──────────────────────────────────────────────────────────────────────────────

def write_dislocation_info(filename, disloc_type, burgers, method, nu,
                            line_dir, plane_normal, n_atoms,
                            energy_before=None, energy_after=None):
    with open(filename, "w") as f:
        f.write("Dislocation structure summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Type             : {disloc_type}\n")
        f.write(f"Burgers magnitude: {burgers:.6f} Å\n")
        f.write(f"Elasticity model : {method}\n")
        if method == "isotropic":
            f.write(f"Poisson ratio    : {nu:.4f}\n")
        f.write(f"Line direction ξ : {np.round(line_dir,  6).tolist()}\n")
        f.write(f"Plane normal n   : {np.round(plane_normal, 6).tolist()}\n")
        f.write(f"N atoms          : {n_atoms}\n")
        if energy_before is not None:
            f.write(f"Energy (before)  : {energy_before:.6f} eV\n")
        if energy_after is not None:
            f.write(f"Energy (after)   : {energy_after:.6f} eV\n")
            dE = energy_after - energy_before
            f.write(f"Relaxation ΔE    : {dE:.6f} eV\n")
    print(f"Summary written to: {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build edge/screw/mixed dislocation structures using the Volterra "
            "displacement field (isotropic) or Stroh anisotropic formalism. "
            "Optionally relax with GRACE."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1],
    )

    # Input/output
    parser.add_argument("--poscar", default="POSCAR", help="Perfect crystal POSCAR/CONTCAR.")
    parser.add_argument("--output", default="POSCAR_dislocation", help="Output POSCAR filename.")
    parser.add_argument("--info",   default="dislocation_info.txt", help="Summary file.")

    # Dislocation parameters
    parser.add_argument("--type", choices=["edge", "screw", "mixed"], default="screw",
                        help="Dislocation type (default: screw).")
    parser.add_argument("--burgers", type=float, required=True,
                        help="Burgers vector magnitude in Å.")
    parser.add_argument("--dipole", action="store_true", default=True,
                        help="Build a dipole (two dislocations of opposite sign, default).")
    parser.add_argument("--single", action="store_true",
                        help="Build a single dislocation instead of a dipole.")

    # Geometry: Cartesian directions
    parser.add_argument("--line",   type=float, nargs=3, default=[0, 0, 1],
                        metavar=("lx", "ly", "lz"),
                        help="Dislocation line direction ξ in Cartesian (default: 0 0 1).")
    parser.add_argument("--normal", type=float, nargs=3, default=[0, 1, 0],
                        metavar=("nx", "ny", "nz"),
                        help="Slip plane normal n in Cartesian (default: 0 1 0).")

    # Geometry: Miller indices (override --line / --normal if given)
    parser.add_argument("--uvw", type=float, nargs=3, metavar=("u", "v", "w"),
                        help="Dislocation line direction as Miller indices [uvw].")
    parser.add_argument("--hkl", type=float, nargs=3, metavar=("h", "k", "l"),
                        help="Slip plane normal as Miller indices (hkl).")

    # Elasticity
    parser.add_argument("--method", choices=["isotropic", "anisotropic"], default="isotropic",
                        help="Displacement field method (default: isotropic).")
    parser.add_argument("--poisson", type=float, default=0.3,
                        help="Poisson ratio for isotropic Volterra (default: 0.3).")
    parser.add_argument("--elastic", default=None,
                        help="Elastic constant file (ELASTIC_Cij_GPa.dat) for anisotropic Stroh.")

    # Relaxation
    parser.add_argument("--relax", action="store_true",
                        help="Relax the dislocation structure with GRACE after building.")
    parser.add_argument("--model", default="GRACE-2L-OAM",
                        help="GRACE model for relaxation (default: GRACE-2L-OAM).")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Force convergence threshold for relaxation in eV/Å (default: 0.05).")
    parser.add_argument("--steps", type=int, default=500,
                        help="Max relaxation steps (default: 500).")

    args = parser.parse_args()

    # ── Load structure ──────────────────────────────────────────────────────
    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    atoms = read(args.poscar, format="vasp")
    print(f"Loaded: {len(atoms)} atoms from {args.poscar}")

    # ── Resolve geometry ────────────────────────────────────────────────────
    cell = np.array(atoms.get_cell())

    line_dir = np.array(args.line, dtype=float)
    normal   = np.array(args.normal, dtype=float)

    if args.uvw is not None:
        line_dir = miller_to_cartesian(args.uvw, cell, reciprocal=False)
        print(f"Miller [uvw]={args.uvw} → Cartesian line direction: {np.round(line_dir, 5)}")

    if args.hkl is not None:
        normal = miller_to_cartesian(args.hkl, cell, reciprocal=True)
        print(f"Miller (hkl)={args.hkl} → Cartesian plane normal: {np.round(normal, 5)}")

    # ── Build dislocation ───────────────────────────────────────────────────
    builder = DislocationBuilder(
        atoms,
        method=args.method,
        poisson_ratio=args.poisson,
        elastic_file=args.elastic,
        line_dir=line_dir,
        plane_normal=normal,
    )

    print(f"\nBuilding {args.type} dislocation  (method={args.method}, b={args.burgers} Å)")
    print(f"  Line direction ξ : {np.round(builder.xi, 5)}")
    print(f"  Plane normal   n : {np.round(builder.n, 5)}")
    print(f"  In-plane       m : {np.round(builder.m, 5)}")

    use_single = args.single and not args.dipole
    if use_single:
        disloc_atoms = builder.build_single(args.type, args.burgers)
    else:
        disloc_atoms = builder.build_dipole(args.type, args.burgers)
        print("  Configuration  : dipole (+b at Lm/4, −b at 3Lm/4)")

    write(args.output, disloc_atoms, format="vasp")
    print(f"\nDislocation structure written to: {args.output}")

    # ── Optional GRACE relaxation ───────────────────────────────────────────
    energy_before = None
    energy_after  = None

    if args.relax:
        from ase.optimize import LBFGS

        if os.path.exists(args.model):
            from tensorpotential.calculator import TPCalculator
            calc = TPCalculator(args.model)
        else:
            from tensorpotential.calculator.foundation_models import grace_fm
            calc = grace_fm(args.model)

        disloc_atoms.calc = calc

        print(f"\nRelaxing with GRACE model: {args.model}")
        print(f"  fmax={args.fmax} eV/Å, max steps={args.steps}")

        energy_before = disloc_atoms.get_potential_energy()
        print(f"  Energy before relaxation : {energy_before:.6f} eV")

        opt = LBFGS(disloc_atoms, trajectory="dislocation_relax.traj",
                    logfile="dislocation_relax.log")
        opt.run(fmax=args.fmax, steps=args.steps)

        energy_after = disloc_atoms.get_potential_energy()
        print(f"  Energy after  relaxation : {energy_after:.6f} eV")
        print(f"  Relaxation ΔE            : {energy_after - energy_before:.6f} eV")

        write(args.output, disloc_atoms, format="vasp")
        print(f"Relaxed structure written to: {args.output}")

    # ── Summary file ────────────────────────────────────────────────────────
    write_dislocation_info(
        args.info,
        disloc_type=args.type,
        burgers=args.burgers,
        method=args.method,
        nu=args.poisson,
        line_dir=builder.xi,
        plane_normal=builder.n,
        n_atoms=len(disloc_atoms),
        energy_before=energy_before,
        energy_after=energy_after,
    )


if __name__ == "__main__":
    main()
