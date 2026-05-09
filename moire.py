#!/usr/bin/env python3
"""
moire.py

Moiré superlattice builder for twisted bilayer systems with GRACE MLIP.

Constructs commensurate twisted-bilayer supercells using the Coincidence Site
Lattice (CSL) method, optionally relaxes with GRACE, and analyses the resulting
stacking pattern.

Physics background
------------------
When two identical 2D layers are stacked with a relative twist angle θ, the
periodic moiré pattern has a large real-space period:

    L_moiré ≈ a / (2 sin(θ/2))

For the supercell to be periodic, the twist must correspond to a commensurate
CSL angle defined by integers (m, n):

    cos θ = (m² + 4mn + n²) / (2(m² + mn + n²))   [hexagonal lattice]
    cos θ = (m² + n²) / (m² + 2mn + n²)            [square lattice]

The CSL supercell contains N = m² + mn + n² (hexagonal) primitive cells per
layer. For square lattices N = m² + n².

Stacking analysis
-----------------
After building (and optionally relaxing) the bilayer, we map each atom in the
top layer to its local stacking configuration relative to the bottom layer:
    AA  — directly above (high energy, unstable)
    AB  — Bernal stacking (honeycomb: most stable)
    SP  — saddle point

Usage
-----
    # Twist graphene at the magic angle (~1.05°): (m,n) = (31,32)
    python moire.py --poscar POSCAR_monolayer --m 5 --n 6

    # Specify interlayer distance and relax with GRACE
    python moire.py --poscar POSCAR --m 5 --n 6 --gap 3.35 --relax --model GRACE-2L-OAM

    # Square lattice
    python moire.py --poscar POSCAR --m 3 --n 2 --lattice square

    # Scan over multiple (m,n) pairs
    python moire.py --poscar POSCAR --scan --m_max 8

Outputs
-------
    POSCAR_moire          commensurate bilayer supercell
    moire_info.txt        CSL parameters, twist angle, moiré period
    stacking_map.dat      x, y, stacking type for each top-layer atom
    stacking_map.png      2D map of stacking regions
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from ase.io import read, write
from ase import Atoms

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# CSL angle calculation
# ──────────────────────────────────────────────────────────────────────────────

def csl_angle_hex(m, n):
    """
    Return the CSL twist angle (degrees) for a hexagonal lattice.
    Valid for m > n > 0, gcd(m,n) ∈ {1} and (m-n) not divisible by 3.

    cos θ = (m² + 4mn + n²) / (2(m² + mn + n²))
    N_cell = m² + mn + n²
    """
    num = m*m + 4*m*n + n*n
    den = 2 * (m*m + m*n + n*n)
    cos_theta = num / den
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_deg = np.degrees(np.arccos(cos_theta))
    N = m*m + m*n + n*n
    L_ratio = 1.0 / (2 * np.sin(np.radians(theta_deg / 2))) if theta_deg > 0 else np.inf
    return theta_deg, N, L_ratio


def csl_angle_square(m, n):
    """
    Return the CSL twist angle (degrees) for a square lattice.
    cos θ = (m² - n²) / (m² + n²),  N = m² + n²
    """
    num = m*m - n*n
    den = m*m + n*n
    if den == 0:
        return 0.0, 1, np.inf
    cos_theta = np.clip(num / den, -1, 1)
    theta_deg = np.degrees(np.arccos(cos_theta))
    N = m*m + n*n
    L_ratio = 1.0 / (2 * np.sin(np.radians(theta_deg / 2))) if theta_deg > 0 else np.inf
    return theta_deg, N, L_ratio


def scan_csl_angles(m_max, lattice="hex", theta_min=0.5, theta_max=30.0):
    """Enumerate commensurate angles for (m,n) with 1 ≤ n < m ≤ m_max."""
    results = []
    for m in range(2, m_max + 1):
        for n in range(1, m):
            if lattice == "hex":
                theta, N, L = csl_angle_hex(m, n)
            else:
                theta, N, L = csl_angle_square(m, n)
            if theta_min <= theta <= theta_max:
                results.append((m, n, theta, N, L))
    results.sort(key=lambda x: x[2])
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Rotation matrix
# ──────────────────────────────────────────────────────────────────────────────

def rotation_matrix_2d(theta_deg):
    """2D rotation matrix (acts on xy plane, z unchanged)."""
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    return np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])


# ──────────────────────────────────────────────────────────────────────────────
# CSL supercell matrix
# ──────────────────────────────────────────────────────────────────────────────

def build_csl_supercell_matrix_hex(m, n):
    """
    Return the 2×2 integer supercell matrix M such that the moiré supercell
    vectors are a_super = M @ a_prim.

    For the hexagonal (m,n) CSL:
        M = [[m, -n],
             [n,  m+n]]
    This gives |det M| = m²+mn+n² = N_cell.
    """
    return np.array([[m, -n], [n, m + n]], dtype=int)


def build_csl_supercell_matrix_square(m, n):
    """Square lattice CSL: M = [[m,-n],[n,m]]."""
    return np.array([[m, -n], [n, m]], dtype=int)


# ──────────────────────────────────────────────────────────────────────────────
# Bilayer builder
# ──────────────────────────────────────────────────────────────────────────────

def make_monolayer_supercell(mono, M):
    """
    Tile the monolayer by the integer 2×2 matrix M.
    Returns an ASE Atoms object.
    """
    from ase.build import make_supercell
    M3 = np.eye(3, dtype=int)
    M3[0, 0] = M[0, 0]; M3[0, 1] = M[0, 1]
    M3[1, 0] = M[1, 0]; M3[1, 1] = M[1, 1]
    sc = make_supercell(mono, M3)
    return sc


def build_bilayer(mono, M, theta_deg, gap, vacuum=15.0, lattice="hex"):
    """
    Construct the twisted bilayer:
        1. Bottom layer: monolayer supercell (no rotation)
        2. Top layer: same supercell, rotated by theta_deg around z-axis
        3. Top layer shifted up by `gap` Å along z
        4. Add vacuum along z

    Returns ASE Atoms for the bilayer.
    """
    bottom = make_monolayer_supercell(mono, M)

    # Rotate top layer
    top = bottom.copy()
    R = rotation_matrix_2d(theta_deg)

    # Rotate positions around centroid (xy plane)
    center = np.array([bottom.cell[0, 0] / 2, bottom.cell[1, 1] / 2, 0.0])
    pos_top = top.positions - center
    pos_top = pos_top @ R.T
    pos_top += center

    # Shift by gap
    pos_top[:, 2] += gap

    top.positions = pos_top

    # Combine
    bilayer = bottom + top

    # Adjust cell z for vacuum
    cell = np.array(bilayer.cell)
    z_max = bilayer.positions[:, 2].max()
    z_min = bilayer.positions[:, 2].min()
    cell[2, 2] = (z_max - z_min) + vacuum
    bilayer.set_cell(cell)

    # Centre in vacuum
    bilayer.positions[:, 2] -= z_min
    bilayer.positions[:, 2] += (vacuum / 2)

    bilayer.set_pbc([True, True, False])
    return bilayer


# ──────────────────────────────────────────────────────────────────────────────
# GRACE relaxation
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path):
    from tensorpotential.calculator.foundation_models import grace_fm
    from tensorpotential.calculator import TPCalculator
    if os.path.exists(model_path):
        return TPCalculator(model_path)
    return grace_fm(model_path)


def relax_bilayer(bilayer, calc, fmax=0.05, steps=300):
    """Relax the bilayer with GRACE, keeping z-cell fixed."""
    from ase.optimize import FIRE
    from ase.constraints import FixedPlane

    bilayer.calc = calc

    # Fix x,y of bottom layer (prevent rigid drift)
    n_bottom = len(bilayer) // 2
    constraints = [FixedPlane(i, [0, 0, 1]) for i in range(n_bottom)]
    bilayer.set_constraint(constraints)

    opt = FIRE(bilayer, logfile="moire_relax.log")
    opt.run(fmax=fmax, steps=steps)
    bilayer.set_constraint()
    return bilayer


# ──────────────────────────────────────────────────────────────────────────────
# Stacking analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyse_stacking(bilayer, n_bottom, a_lattice):
    """
    For each top-layer atom, find the nearest bottom-layer atom and the
    lateral displacement vector. Classify as AA, AB, or SP based on
    |displacement| relative to the lattice constant.

    Returns list of (x, y, label) for each top-layer atom.
    """
    pos = bilayer.positions
    bot_pos = pos[:n_bottom]
    top_pos = pos[n_bottom:]

    cell = np.array(bilayer.cell)

    stacking = []
    for tp in top_pos:
        dv = tp[None, :2] - bot_pos[:, :2]
        # Minimum image in xy (approximate, ignores non-orthorhombic)
        dv -= np.round(dv / np.diag(cell[:2, :2])) * np.diag(cell[:2, :2])
        d = np.linalg.norm(dv, axis=1)
        d_min = d.min()

        # Classify
        if d_min < 0.15 * a_lattice:
            label = "AA"
        elif d_min < 0.65 * a_lattice:
            label = "AB"
        else:
            label = "SP"

        stacking.append((tp[0], tp[1], label))

    return stacking


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def write_moire_info(m, n, theta_deg, N_cell, L_ratio, a_lattice, lattice,
                     n_atoms, filename="moire_info.txt"):
    L_moire_A = L_ratio * a_lattice
    with open(filename, "w") as f:
        f.write("Moiré superlattice information\n")
        f.write("=" * 40 + "\n")
        f.write(f"Lattice type         : {lattice}\n")
        f.write(f"CSL indices          : m={m}, n={n}\n")
        f.write(f"Twist angle θ        : {theta_deg:.6f}°\n")
        f.write(f"CSL cells per layer  : {N_cell}\n")
        f.write(f"Moiré period (L/a)   : {L_ratio:.4f}\n")
        f.write(f"Moiré period (Å)     : {L_moire_A:.4f}\n")
        f.write(f"Total atoms          : {n_atoms}\n")
    print(f"Wrote {filename}")


def write_stacking_map(stacking, filename="stacking_map.dat"):
    with open(filename, "w") as f:
        f.write(f"{'#x(A)':>12}  {'y(A)':>12}  {'stacking':>10}\n")
        for x, y, lbl in stacking:
            f.write(f"{x:12.6f}  {y:12.6f}  {lbl:>10}\n")
    print(f"Wrote {filename}  ({len(stacking)} top-layer atoms)")


def plot_stacking_map(stacking, cell, theta_deg, filename="stacking_map.png"):
    colors = {"AA": "red", "AB": "steelblue", "SP": "gold"}
    labels_done = set()

    fig, ax = plt.subplots(figsize=(6, 6))
    for x, y, lbl in stacking:
        c = colors.get(lbl, "gray")
        l = lbl if lbl not in labels_done else None
        ax.scatter(x, y, c=c, s=10, alpha=0.8, label=l)
        labels_done.add(lbl)

    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(f"Moiré stacking map  θ = {theta_deg:.3f}°")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Moiré superlattice builder for twisted bilayers with GRACE."
    )
    parser.add_argument("--poscar",  default="POSCAR",
                        help="Monolayer POSCAR (single layer).")
    parser.add_argument("--m",       type=int, default=5,
                        help="CSL integer m (default: 5).")
    parser.add_argument("--n",       type=int, default=6,
                        help="CSL integer n (default: 6).")
    parser.add_argument("--lattice", default="hex", choices=["hex", "square"],
                        help="Lattice symmetry for CSL angle formula (default: hex).")
    parser.add_argument("--gap",     type=float, default=3.35,
                        help="Interlayer distance in Å (default: 3.35).")
    parser.add_argument("--vacuum",  type=float, default=20.0,
                        help="Vacuum thickness in Å (default: 20.0).")
    parser.add_argument("--relax",   action="store_true",
                        help="Relax with GRACE after building.")
    parser.add_argument("--model",   default="GRACE-2L-OAM",
                        help="GRACE model (used with --relax).")
    parser.add_argument("--fmax",    type=float, default=0.05,
                        help="Force convergence in eV/Å (default: 0.05).")
    parser.add_argument("--scan",    action="store_true",
                        help="Print table of commensurate angles and exit.")
    parser.add_argument("--m_max",   type=int, default=10,
                        help="Upper limit for m in --scan mode (default: 10).")
    parser.add_argument("--theta_min", type=float, default=0.5,
                        help="Min angle for --scan (default: 0.5°).")
    parser.add_argument("--theta_max", type=float, default=30.0,
                        help="Max angle for --scan (default: 30.0°).")
    args = parser.parse_args()

    # ── Scan mode ──────────────────────────────────────────────────────────────
    if args.scan:
        results = scan_csl_angles(args.m_max, args.lattice,
                                  args.theta_min, args.theta_max)
        print(f"\nCommensurate CSL angles ({args.lattice} lattice, "
              f"{args.theta_min}° < θ < {args.theta_max}°):\n")
        print(f"{'m':>5}  {'n':>5}  {'θ (°)':>10}  {'N_cells':>8}  {'L_moiré/a':>12}")
        print("-" * 50)
        for m, n, theta, N, L in results:
            print(f"{m:5d}  {n:5d}  {theta:10.4f}  {N:8d}  {L:12.4f}")
        print(f"\nTotal: {len(results)} commensurate angles found.")
        return

    # ── Build mode ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    mono = read(args.poscar, format="vasp")
    cell = np.array(mono.get_cell())
    a_lattice = np.linalg.norm(cell[0])  # in-plane lattice constant

    if args.lattice == "hex":
        theta_deg, N_cell, L_ratio = csl_angle_hex(args.m, args.n)
        M = build_csl_supercell_matrix_hex(args.m, args.n)
    else:
        theta_deg, N_cell, L_ratio = csl_angle_square(args.m, args.n)
        M = build_csl_supercell_matrix_square(args.m, args.n)

    print(f"CSL ({args.m},{args.n}) → θ = {theta_deg:.5f}°, "
          f"N_cell = {N_cell}, L_moiré ≈ {L_ratio * a_lattice:.2f} Å")
    print(f"Building moiré supercell...")

    bilayer = build_bilayer(mono, M, theta_deg, args.gap,
                            vacuum=args.vacuum, lattice=args.lattice)

    n_bottom = len(bilayer) // 2
    print(f"Bilayer: {len(bilayer)} atoms  ({n_bottom} per layer)")

    if args.relax:
        print(f"\nRelaxing with {args.model}  (fmax={args.fmax} eV/Å)...")
        calc = load_model(args.model)
        bilayer = relax_bilayer(bilayer, calc, fmax=args.fmax)

    write("POSCAR_moire", bilayer, format="vasp")
    print("Wrote POSCAR_moire")

    stacking = analyse_stacking(bilayer, n_bottom, a_lattice)
    write_stacking_map(stacking)
    plot_stacking_map(stacking, np.array(bilayer.cell), theta_deg)
    write_moire_info(args.m, args.n, theta_deg, N_cell, L_ratio,
                     a_lattice, args.lattice, len(bilayer))


if __name__ == "__main__":
    main()
