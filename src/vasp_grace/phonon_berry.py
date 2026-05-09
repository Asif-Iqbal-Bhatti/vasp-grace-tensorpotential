#!/usr/bin/env python3
"""
phonon_berry.py

Phonon Zak phase and Berry curvature for 1D k-paths with GRACE MLIP.

Computes the Zak phase (Berry phase along a full Brillouin zone path) for
each phonon band, providing a Z₂ topological invariant for phononic systems.

Physics background
------------------
The Zak phase for band n along a closed path in reciprocal space:

    γₙ = -Im[ ln ∏_{j=0}^{N-1} ⟨uₙ(kⱼ) | uₙ(kⱼ₊₁)⟩ ]

where |uₙ(k)⟩ is the periodic part of the Bloch-like phonon eigenvector.
For a 1D BZ path (Γ → X → Γ or Γ → Γ across the BZ):
    γ = 0  → trivial (even winding)
    γ = π  → topological (odd winding, band inversion present)

Physical meaning:
    - Phonon Zak phase classifies the topological character of each branch
    - γ = π signals a phononic analogue of a topological insulator band
    - Band-resolved Zak phases can reveal polarization, charge pumping,
      and phononic edge states in finite systems

This module also computes local Berry curvature Ωₙ(k) = Im[∂kₓ uₙ†  ∂ky uₙ]
using finite differences, which integrates to give the Chern number for 2D BZs.

Usage
-----
    # Zak phase for all bands along Γ-X-Γ path
    python phonon_berry.py --poscar POSCAR --model GRACE-2L-OAM

    # Custom path and resolution
    python phonon_berry.py --poscar POSCAR --model GRACE-2L-OAM \\
                           --path GXG --nkpts 200

    # Compute Berry curvature on a 2D kx-ky mesh
    python phonon_berry.py --poscar POSCAR --model GRACE-2L-OAM \\
                           --curvature --nkx 30 --nky 30

Outputs
-------
    phonon_zak.dat          band index, Zak phase (rad), Zak phase (°), class
    phonon_zak.png          bar chart of Zak phases per band
    phonon_berry_curv.dat   kx, ky, band, Ω(k) [if --curvature]
    phonon_chern.dat        band, Chern number (integrated Ω) [if --curvature]
    phonon_chern.png        Berry curvature map [if --curvature]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ase.io import read
from ase.phonons import Phonons
from ase import units


# ──────────────────────────────────────────────────────────────────────────────
# Model + phonon setup
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path):
    from tensorpotential.calculator.foundation_models import grace_fm
    from tensorpotential.calculator import TPCalculator
    if os.path.exists(model_path):
        return TPCalculator(model_path)
    return grace_fm(model_path)


def build_phonon_object(atoms, calc, supercell=(3, 3, 3), delta=0.015):
    """Load cached phonon data (phonon.*.json) or run fresh displacements."""
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta, name="phonon")
    cache_exists = os.path.exists(f"phonon.{0}.{0}.json")
    if cache_exists:
        print("Loading cached phonon force constants (phonon.*.json)...")
        ph.read(method="Frederiksen", symmetrize=3, acoustic=True,
                cutoff=None, born=False)
    else:
        print("No cached data. Running finite-displacement calculations...")
        ph.run()
        ph.read(method="Frederiksen", symmetrize=3, acoustic=True,
                cutoff=None, born=False)
    return ph


# ──────────────────────────────────────────────────────────────────────────────
# BZ path builder
# ──────────────────────────────────────────────────────────────────────────────

def build_bz_path(atoms, path_str, n_kpts):
    """
    Build a k-path for Zak phase calculation.
    The path should close the BZ (e.g. 'GXG' for 1D).

    Returns q_pts (N, 3) and the bandpath object.
    """
    path = atoms.cell.bandpath(path_str, npoints=n_kpts)
    return path.kpts, path


def get_eigenvectors(ph, q_pts):
    """
    Compute phonon eigenvectors at all q_pts.
    Returns omega (N, n_modes) and u (N, n_modes, n_modes) or None.
    """
    print(f"Computing phonon eigenvectors at {len(q_pts)} k-points...")
    try:
        omega_raw, u_kn = ph.band_structure(q_pts, modes=True)
    except TypeError:
        print("Warning: this ASE version does not support modes=True. "
              "Zak phase calculation requires eigenvectors.")
        omega_raw = ph.band_structure(q_pts)
        return omega_raw, None

    ev_to_THz = units._e / (2 * np.pi * units._hbar) * 1e-12
    omega = np.sign(omega_raw) * np.sqrt(np.abs(omega_raw)) * ev_to_THz
    return omega, u_kn


# ──────────────────────────────────────────────────────────────────────────────
# Zak phase
# ──────────────────────────────────────────────────────────────────────────────

def compute_zak_phases(u_kn):
    """
    Compute Zak phase for each phonon band via Wilson loop along the k-path.

    γₙ = -Im[ ln ∏_{j=0}^{N-2} ⟨uₙ(kⱼ) | uₙ(kⱼ₊₁)⟩ ]

    The path must be closed (first and last k-point are equivalent under a
    reciprocal lattice vector, e.g. Γ→X→Γ or Γ→Γ traversing the BZ).

    Parameters
    ----------
    u_kn : (N_kpts, n_modes, n_modes)  eigenvectors

    Returns
    -------
    zak : (n_modes,)  Zak phase in radians  ∈ (-π, π]
    """
    n_kpts, n_modes = u_kn.shape[0], u_kn.shape[1]
    zak = np.zeros(n_modes)

    for band in range(n_modes):
        product = 1.0 + 0j
        for k in range(n_kpts - 1):
            uk  = u_kn[k,     band, :]
            uk1 = u_kn[k + 1, band, :]
            overlap = np.vdot(uk, uk1)   # ⟨uk|uk1⟩
            if abs(overlap) > 1e-12:
                product *= overlap / abs(overlap)
        zak[band] = -np.imag(np.log(product))

    return zak


def classify_zak(gamma):
    """Classify Zak phase: trivial (≈0) or topological (≈π)."""
    g = abs(gamma) % (2 * np.pi)
    if g < 0.3 or g > 2 * np.pi - 0.3:
        return "trivial (γ≈0)"
    elif abs(g - np.pi) < 0.3:
        return "topological (γ≈π)"
    else:
        return f"intermediate (γ={np.degrees(gamma):.1f}°)"


# ──────────────────────────────────────────────────────────────────────────────
# Berry curvature on a 2D k-mesh
# ──────────────────────────────────────────────────────────────────────────────

def build_2d_kmesh(atoms, nkx, nky, kz=0.0):
    """
    Build a uniform kx-ky mesh in the first BZ.
    Returns (nkx*nky, 3) array of k-points in fractional coords.
    """
    kx = np.linspace(0, 1, nkx, endpoint=False)
    ky = np.linspace(0, 1, nky, endpoint=False)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    q = np.zeros((nkx * nky, 3))
    q[:, 0] = KX.ravel()
    q[:, 1] = KY.ravel()
    q[:, 2] = kz
    return q, KX, KY


def compute_berry_curvature_2d(ph, atoms, nkx, nky, dk=0.005):
    """
    Compute Berry curvature Ωₙ(k) = Im[∂kx uₙ† ∂ky uₙ] using finite differences.

    Ωₙ(k) ≈ Im[ ⟨∂kx uₙ | ∂ky uₙ⟩ ]

    Approximated via:
        ∂kx uₙ(k) ≈ [uₙ(k+dk_x) - uₙ(k-dk_x)] / (2 dk)

    Returns
    -------
    Omega : (n_modes, nkx, nky) Berry curvature
    """
    q, KX, KY = build_2d_kmesh(atoms, nkx, nky)

    print(f"Computing Berry curvature on {nkx}×{nky} k-mesh "
          f"({len(q)} k-points × 4 finite-difference shifts)...")

    # We need 4 additional meshes for finite differences
    def shifted_mesh(dkx=0, dky=0):
        q_shifted = q.copy()
        q_shifted[:, 0] += dkx
        q_shifted[:, 1] += dky
        return q_shifted

    q_px = shifted_mesh(dkx=+dk)
    q_mx = shifted_mesh(dkx=-dk)
    q_py = shifted_mesh(dky=+dk)
    q_my = shifted_mesh(dky=-dk)

    def get_u(qpts):
        try:
            _, u = ph.band_structure(qpts, modes=True)
            return u   # (N, n_modes, n_modes)
        except Exception:
            return None

    u_px = get_u(q_px)
    u_mx = get_u(q_mx)
    u_py = get_u(q_py)
    u_my = get_u(q_my)

    if u_px is None:
        print("Error: eigenvectors unavailable — Berry curvature skipped.")
        return None

    n_modes = u_px.shape[1]
    N = len(q)

    # Central difference derivatives
    du_dkx = (u_px - u_mx) / (2 * dk)   # (N, n_modes, n_modes)
    du_dky = (u_py - u_my) / (2 * dk)

    Omega = np.zeros((n_modes, nkx, nky))
    for band in range(n_modes):
        for i in range(N):
            ix = i // nky
            iy = i % nky
            dx = du_dkx[i, band, :]
            dy = du_dky[i, band, :]
            Omega[band, ix, iy] = np.imag(np.vdot(dx, dy))

    return Omega


def compute_chern_numbers(Omega, nkx, nky):
    """
    Chern number = (1/2π) ∫∫_BZ Ωₙ(k) dkx dky
    Approximated by trapezoidal sum over the mesh.
    """
    dkx = 1.0 / nkx
    dky = 1.0 / nky
    n_modes = Omega.shape[0]
    chern = np.zeros(n_modes)
    for band in range(n_modes):
        chern[band] = np.sum(Omega[band]) * dkx * dky / (2 * np.pi)
    return chern


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def write_zak(zak, filename="phonon_zak.dat"):
    with open(filename, "w") as f:
        f.write(f"{'#band':>6}  {'zak(rad)':>12}  {'zak(deg)':>12}  {'class':>30}\n")
        for n, g in enumerate(zak):
            cls = classify_zak(g)
            f.write(f"{n:6d}  {g:12.6f}  {np.degrees(g):12.4f}  {cls:>30}\n")
    print(f"Wrote {filename}")


def plot_zak(zak, omega_at_gamma, filename="phonon_zak.png"):
    """Bar chart: Zak phase per band, colour-coded by trivial/topological."""
    n_modes = len(zak)
    colors = []
    for g in zak:
        gm = abs(g) % (2 * np.pi)
        if abs(gm - np.pi) < 0.3:
            colors.append("tomato")
        elif gm < 0.3 or gm > 2 * np.pi - 0.3:
            colors.append("steelblue")
        else:
            colors.append("gold")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.bar(range(n_modes), np.degrees(zak), color=colors, edgecolor="k", lw=0.4)
    ax.axhline(180, ls="--", c="tomato", lw=0.8, label="γ = π (topological)")
    ax.axhline(0,   ls="--", c="steelblue", lw=0.8, label="γ = 0 (trivial)")
    ax.set_xlabel("Band index")
    ax.set_ylabel("Zak phase (°)")
    ax.set_title("Phonon Zak phases")
    ax.legend(fontsize=8)

    # Frequency at Γ vs Zak phase scatter
    ax2 = axes[1]
    sc = ax2.scatter(np.degrees(zak), omega_at_gamma, c=np.degrees(zak),
                     cmap="RdBu_r", vmin=-190, vmax=190, s=30)
    plt.colorbar(sc, ax=ax2, label="Zak phase (°)")
    ax2.set_xlabel("Zak phase (°)")
    ax2.set_ylabel("Frequency at Γ (THz)")
    ax2.set_title("Frequency vs Zak phase")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


def write_curvature(Omega, nkx, nky, filename="phonon_berry_curv.dat"):
    n_modes = Omega.shape[0]
    with open(filename, "w") as f:
        f.write(f"{'#kx_frac':>10}  {'ky_frac':>10}  {'band':>6}  {'Omega':>14}\n")
        for ix in range(nkx):
            for iy in range(nky):
                kx = ix / nkx
                ky = iy / nky
                for band in range(n_modes):
                    f.write(f"{kx:10.6f}  {ky:10.6f}  {band:6d}  "
                            f"{Omega[band, ix, iy]:14.8f}\n")
    print(f"Wrote {filename}")


def write_chern(chern, filename="phonon_chern.dat"):
    with open(filename, "w") as f:
        f.write(f"{'#band':>6}  {'Chern':>12}  {'Chern_int':>12}\n")
        for n, c in enumerate(chern):
            f.write(f"{n:6d}  {c:12.6f}  {int(round(c)):12d}\n")
    print(f"Wrote {filename}")


def plot_chern(Omega, chern, nkx, nky, n_bands_show=6, filename="phonon_chern.png"):
    """Plot Berry curvature map for a few bands."""
    show = min(n_bands_show, Omega.shape[0])
    cols = 3
    rows = (show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).ravel()

    kx = np.linspace(0, 1, nkx, endpoint=False)
    ky = np.linspace(0, 1, nky, endpoint=False)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    for b in range(show):
        ax = axes[b]
        data = Omega[b]
        vmax = max(abs(data.max()), abs(data.min()), 1e-10)
        pcm = ax.pcolormesh(KX, KY, data, cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax, shading="auto")
        plt.colorbar(pcm, ax=ax, label="Ω(k)")
        ax.set_title(f"Band {b}  C={chern[b]:.2f}", fontsize=9)
        ax.set_xlabel("kx"); ax.set_ylabel("ky")

    for b in range(show, len(axes)):
        axes[b].set_visible(False)

    fig.suptitle("Phonon Berry curvature", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


def write_summary(zak, chern, n_modes, n_imaginary, filename="phonon_berry_summary.txt"):
    n_topo = sum(1 for g in zak if abs(abs(g) % (2*np.pi) - np.pi) < 0.3)
    with open(filename, "w") as f:
        f.write("Phonon Berry phase / Zak phase summary\n")
        f.write("=" * 42 + "\n")
        f.write(f"Total phonon modes     : {n_modes}\n")
        f.write(f"Imaginary modes        : {n_imaginary}\n")
        f.write(f"Topological bands (γ≈π): {n_topo}\n")
        if chern is not None:
            n_nontriv = sum(1 for c in chern if abs(round(c)) != 0)
            f.write(f"Non-zero Chern bands   : {n_nontriv}\n")
        f.write("\nBand-resolved Zak phases:\n")
        f.write(f"  {'band':>5}  {'γ (°)':>10}  {'class':>30}\n")
        for n, g in enumerate(zak):
            f.write(f"  {n:5d}  {np.degrees(g):10.2f}  {classify_zak(g):>30}\n")
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phonon Zak phase and Berry curvature with GRACE MLIP."
    )
    parser.add_argument("--poscar",    default="POSCAR",       help="Unit cell POSCAR.")
    parser.add_argument("--model",     default="GRACE-2L-OAM", help="GRACE model.")
    parser.add_argument("--path",      default="GXG",
                        help="k-path string (default: GXG — closed BZ path for Zak).")
    parser.add_argument("--nkpts",     type=int, default=200,
                        help="k-points along path (default: 200).")
    parser.add_argument("--supercell", type=int, nargs=3, default=[3, 3, 3],
                        metavar=("Nx", "Ny", "Nz"),
                        help="Phonon supercell (default: 3 3 3).")
    parser.add_argument("--delta",     type=float, default=0.015,
                        help="Displacement amplitude in Å (default: 0.015).")
    parser.add_argument("--curvature", action="store_true",
                        help="Compute Berry curvature Ω(k) on a 2D kx-ky mesh.")
    parser.add_argument("--nkx",       type=int, default=20,
                        help="k-points along kx for curvature mesh (default: 20).")
    parser.add_argument("--nky",       type=int, default=20,
                        help="k-points along ky for curvature mesh (default: 20).")
    parser.add_argument("--dk",        type=float, default=0.005,
                        help="Finite-difference step for ∂k (default: 0.005).")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    atoms = read(args.poscar, format="vasp")
    calc  = load_model(args.model)
    ph    = build_phonon_object(atoms, calc, tuple(args.supercell), args.delta)

    # ── Zak phase ──────────────────────────────────────────────────────────────
    q_pts, bandpath = build_bz_path(atoms, args.path, args.nkpts)
    omega, u_kn = get_eigenvectors(ph, q_pts)

    n_imaginary = int(np.sum(omega < 0))
    n_modes     = omega.shape[1]
    print(f"Modes: {n_modes}  |  Imaginary: {n_imaginary}")

    if u_kn is None:
        print("Cannot compute Zak phases without eigenvectors. Exiting.")
        sys.exit(1)

    print(f"\nComputing Zak phases for {n_modes} bands...")
    zak = compute_zak_phases(u_kn)

    # Print summary table
    print(f"\n{'Band':>5}  {'Zak (°)':>10}  {'Classification':>30}")
    print("-" * 50)
    for n, g in enumerate(zak):
        print(f"{n:5d}  {np.degrees(g):10.2f}  {classify_zak(g):>30}")

    # Frequencies at Γ for scatter plot
    gamma_idx = np.argmin(np.linalg.norm(q_pts, axis=1))
    omega_gamma = omega[gamma_idx]

    write_zak(zak)
    plot_zak(zak, omega_gamma)

    # ── Berry curvature (optional) ─────────────────────────────────────────────
    chern = None
    if args.curvature:
        print(f"\nComputing Berry curvature on {args.nkx}×{args.nky} mesh...")
        Omega = compute_berry_curvature_2d(ph, atoms, args.nkx, args.nky, dk=args.dk)
        if Omega is not None:
            chern = compute_chern_numbers(Omega, args.nkx, args.nky)
            print("\nChern numbers (rounded):")
            for n, c in enumerate(chern):
                print(f"  Band {n:3d}: C = {c:.4f}  ≈  {int(round(c))}")
            write_curvature(Omega, args.nkx, args.nky)
            write_chern(chern)
            plot_chern(Omega, chern, args.nkx, args.nky)

    write_summary(zak, chern, n_modes, n_imaginary)


if __name__ == "__main__":
    main()
