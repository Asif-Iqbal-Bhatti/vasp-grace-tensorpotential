#!/usr/bin/env python3
"""
topological_phonons.py

Topological analysis of phonon band structures computed with GRACE MLIP.

Detects band crossings (potential Weyl/Dirac phonon points), computes the
local Berry curvature around each crossing, and classifies phononic topology.
Builds on the existing phonon module (main.py, IBRION=5/6) — reads cached
force constant data (phonon.*.json files).

Physics background
------------------
The phonon band structure can carry non-trivial topological invariants:
    - Weyl phonons   : point-like band crossings with non-zero Chern number
                       (chiral phonons, topological surface arcs)
    - Nodal-line phonons : line of band crossings forming a loop in k-space
    - Dirac phonons  : fourfold-degenerate crossing (two Weyl of opposite chirality)

These are characterized by the Berry phase accumulated along a closed loop
encircling the crossing in k-space:
    - Chern number C = ±1 → Weyl phonon
    - Berry phase γ = π  → topological nodal line

Usage
-----
    # Uses cached phonon data (phonon.*.json) from main.py
    python topological_phonons.py --poscar POSCAR --model GRACE-2L-OAM

    # Custom k-path and crossing threshold
    python topological_phonons.py --poscar POSCAR --model GRACE-2L-OAM \\
                                  --path GXMGZ --nkpts 300 --tol 0.5

    # Compute Berry curvature at each detected crossing
    python topological_phonons.py --poscar POSCAR --model GRACE-2L-OAM --berry

Outputs
-------
    phonon_topo_bands.png     band structure with crossings highlighted
    phonon_crossings.dat      table of detected crossings (k, bands, gap, Berry phase)
    phonon_topo_summary.txt   summary of topological classification
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
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path):
    from tensorpotential.calculator.foundation_models import grace_fm
    from tensorpotential.calculator import TPCalculator
    if os.path.exists(model_path):
        return TPCalculator(model_path)
    return grace_fm(model_path)


# ──────────────────────────────────────────────────────────────────────────────
# Phonon setup
# ──────────────────────────────────────────────────────────────────────────────

def build_phonon_object(atoms, calc, supercell=(3, 3, 3), delta=0.015):
    """
    Initialise ASE Phonons from cached displacement data (phonon.*.json).
    If no cached data is found, runs fresh finite-displacement calculations.
    """
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta, name="phonon")

    cache_exists = os.path.exists(f"phonon.{0}.{0}.json")
    if cache_exists:
        print("Loading cached phonon force constants (phonon.*.json)...")
        ph.read(method="Frederiksen", symmetrize=3, acoustic=True,
                cutoff=None, born=False)
    else:
        print("No cached phonon data found. Running finite-displacement calculations...")
        ph.run()
        ph.read(method="Frederiksen", symmetrize=3, acoustic=True,
                cutoff=None, born=False)
    return ph


# ──────────────────────────────────────────────────────────────────────────────
# Band structure + eigenvectors
# ──────────────────────────────────────────────────────────────────────────────

def get_bands_and_modes(ph, atoms, path_str, n_kpts):
    """
    Compute phonon frequencies and eigenvectors along a k-path.

    Returns
    -------
    q_pts    : (N, 3) k-points in fractional coordinates
    omega_kn : (N, n_modes) frequencies in THz
    u_kn     : list of (n_modes, n_modes) eigenvector matrices
               u_kn[k][n, :] = eigenvector for band n at k-point k
    x_coords : (N,) x-axis coordinates for plotting
    special_x: list of (x, label) for high-symmetry points
    """
    path = atoms.cell.bandpath(path_str, npoints=n_kpts)
    q_pts = path.kpts

    print(f"Computing phonon eigenvectors at {len(q_pts)} k-points...")
    try:
        # ASE internal method that returns eigenvectors
        omega_kn_raw, u_kn = ph.band_structure(q_pts, modes=True)
    except TypeError:
        # Older ASE: modes parameter not supported
        print("Warning: this ASE version does not support modes=True. "
              "Eigenvectors unavailable — Berry phase skipped.")
        omega_kn_raw = ph.band_structure(q_pts)
        u_kn = None

    # Convert ω² (eV²) → THz
    ev_to_THz = units._e / (2 * np.pi * units._hbar) * 1e-12
    omega_kn = np.sign(omega_kn_raw) * np.sqrt(np.abs(omega_kn_raw)) * ev_to_THz

    # Build x-axis from path
    x_coords, special_x = _build_x_axis(path, q_pts, atoms)

    return q_pts, omega_kn, u_kn, x_coords, special_x


def _build_x_axis(path, q_pts, atoms):
    """Build cumulative distance x-axis for band structure plot."""
    rec = np.array(atoms.cell.reciprocal())
    q_cart = q_pts @ rec
    dq = np.linalg.norm(np.diff(q_cart, axis=0), axis=1)
    x = np.concatenate([[0], np.cumsum(dq)])

    # Special points from path
    special_x = []
    if hasattr(path, 'special_points') and path.special_points:
        # Try to get high-symmetry labels from path object
        try:
            for label, kpt in path.special_points.items():
                kpt_cart = np.array(kpt) @ rec
                dists = np.linalg.norm(q_cart - kpt_cart[None, :], axis=1)
                idx = np.argmin(dists)
                special_x.append((x[idx], label))
        except Exception:
            pass

    return x, special_x


# ──────────────────────────────────────────────────────────────────────────────
# Band crossing detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_crossings(omega_kn, x_coords, tol_THz=0.5):
    """
    Detect band crossings: adjacent bands approaching within `tol_THz` THz.

    Two complementary criteria:
    1. Near-degeneracy: |ω_n(k) - ω_{n+1}(k)| < tol
    2. Band inversion: gap changes sign between adjacent k-points

    Returns list of dicts with keys: k_idx, x, band_lo, band_hi, gap, type
    """
    n_kpts, n_modes = omega_kn.shape
    crossings = []
    seen = set()

    for k in range(1, n_kpts - 1):
        for n in range(n_modes - 1):
            gap_k   = omega_kn[k,     n + 1] - omega_kn[k,     n]
            gap_km1 = omega_kn[k - 1, n + 1] - omega_kn[k - 1, n]

            # Near-degeneracy at this k
            near_degen = abs(gap_k) < tol_THz

            # Sign change in gap between k-1 and k (band inversion)
            inversion = gap_k * gap_km1 < 0

            if near_degen or inversion:
                key = (k, n)
                if key not in seen:
                    seen.add(key)
                    crossing_type = "inversion" if inversion else "degeneracy"
                    crossings.append({
                        "k_idx"   : k,
                        "x"       : x_coords[k],
                        "band_lo" : n,
                        "band_hi" : n + 1,
                        "gap"     : abs(gap_k),
                        "type"    : crossing_type,
                        "berry"   : None,     # filled later if --berry
                    })

    # Remove duplicates (same crossing detected at multiple k shifts)
    merged = _merge_nearby_crossings(crossings, x_coords)
    return merged


def _merge_nearby_crossings(crossings, x_coords, merge_window=3):
    """Merge crossings within `merge_window` k-points of each other."""
    if not crossings:
        return []
    merged, used = [], set()
    for i, c in enumerate(crossings):
        if i in used:
            continue
        group = [c]
        for j, c2 in enumerate(crossings[i+1:], start=i+1):
            if j not in used and c2["band_lo"] == c["band_lo"]:
                if abs(c2["k_idx"] - c["k_idx"]) <= merge_window:
                    group.append(c2)
                    used.add(j)
        # Keep the one with smallest gap
        best = min(group, key=lambda g: g["gap"])
        merged.append(best)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Local Berry phase around a crossing
# ──────────────────────────────────────────────────────────────────────────────

def compute_berry_phase_at_crossing(ph, atoms, q0_frac, band_n, n_loop=20, radius=0.02):
    """
    Compute Berry phase γ for phonon band `band_n` on a small closed loop
    in the qx-qy plane centred at q0_frac.

    γ = 0   → trivial (no topological charge)
    γ = π   → topological (Weyl/nodal line)

    Returns the Berry phase in radians.
    """
    # Build a small circular loop in reduced coordinates
    angles = np.linspace(0, 2 * np.pi, n_loop, endpoint=False)
    loop_kpts = np.array([
        q0_frac + radius * np.array([np.cos(a), np.sin(a), 0.0])
        for a in angles
    ])
    # Close the loop
    loop_kpts = np.vstack([loop_kpts, loop_kpts[0]])

    try:
        _, u_loop = ph.band_structure(loop_kpts, modes=True)
    except Exception:
        return None

    # Wilson loop product
    product = 1.0 + 0j
    n_pts = len(u_loop) - 1
    for j in range(n_pts):
        uj  = u_loop[j][band_n, :]
        uj1 = u_loop[j + 1][band_n, :]
        overlap = np.vdot(uj, uj1)
        if abs(overlap) > 1e-10:
            product *= overlap / abs(overlap)

    gamma = -np.imag(np.log(product))
    return gamma


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_crossing(berry_phase):
    """Classify based on Berry phase (radians)."""
    if berry_phase is None:
        return "unknown"
    gamma_mod = abs(berry_phase) % (2 * np.pi)
    if gamma_mod < 0.3 or gamma_mod > 2 * np.pi - 0.3:
        return "trivial"
    elif abs(gamma_mod - np.pi) < 0.3:
        return "Weyl/nodal-line (γ≈π)"
    else:
        return f"non-trivial (γ={np.degrees(berry_phase):.1f}°)"


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

def plot_bands_with_crossings(x_coords, omega_kn, crossings, special_x,
                               filename="phonon_topo_bands.png"):
    n_modes = omega_kn.shape[1]
    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot all bands
    for n in range(n_modes):
        ax.plot(x_coords, omega_kn[:, n], "b-", lw=0.8, alpha=0.6)

    # Highlight crossings
    colors = {"inversion": "red", "degeneracy": "orange"}
    labels = {"inversion": "Band inversion", "degeneracy": "Near-degeneracy"}
    plotted = set()
    for c in crossings:
        ctype = c["type"]
        lbl = labels[ctype] if ctype not in plotted else None
        ax.axvline(c["x"], color=colors[ctype], lw=1.2, alpha=0.7, ls="--", label=lbl)
        n_lo, n_hi = c["band_lo"], c["band_hi"]
        y_mid = 0.5 * (omega_kn[c["k_idx"], n_lo] + omega_kn[c["k_idx"], n_hi])
        berry = c.get("berry")
        marker = "★" if berry is not None and abs(abs(berry) % (2*np.pi) - np.pi) < 0.3 else "×"
        ax.text(c["x"], y_mid, marker, ha="center", va="center",
                fontsize=10, color=colors[ctype])
        plotted.add(ctype)

    # High-symmetry labels
    for x_sp, label in special_x:
        ax.axvline(x_sp, color="k", lw=0.5, alpha=0.4)
        ax.text(x_sp, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1,
                label, ha="center", va="top", fontsize=9)

    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_xlabel("Wave vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Phonon band structure — topological analysis")
    if plotted:
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


def write_crossings(crossings, q_pts, filename="phonon_crossings.dat"):
    with open(filename, "w") as f:
        f.write("# Phonon band crossings detected\n")
        f.write(f"{'#idx':>5}  {'qx':>8}  {'qy':>8}  {'qz':>8}  "
                f"{'band_lo':>8}  {'band_hi':>8}  {'gap(THz)':>10}  "
                f"{'type':>12}  {'berry(rad)':>12}  {'class':>25}\n")
        for i, c in enumerate(crossings):
            q = q_pts[c["k_idx"]]
            berry = c["berry"] if c["berry"] is not None else float("nan")
            cls = classify_crossing(c["berry"])
            f.write(f"{i:5d}  {q[0]:8.5f}  {q[1]:8.5f}  {q[2]:8.5f}  "
                    f"{c['band_lo']:8d}  {c['band_hi']:8d}  {c['gap']:10.4f}  "
                    f"{c['type']:>12}  {berry:12.5f}  {cls:>25}\n")
    print(f"Wrote {filename}  ({len(crossings)} crossings)")


def write_summary(crossings, n_modes, n_imaginary, filename="phonon_topo_summary.txt"):
    n_weyl = sum(1 for c in crossings
                 if c["berry"] is not None
                 and abs(abs(c["berry"]) % (2*np.pi) - np.pi) < 0.3)
    with open(filename, "w") as f:
        f.write("Topological phonon analysis summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total phonon modes      : {n_modes}\n")
        f.write(f"Imaginary modes (< 0)   : {n_imaginary}\n")
        f.write(f"Band crossings detected : {len(crossings)}\n")
        f.write(f"  - by band inversion   : {sum(1 for c in crossings if c['type']=='inversion')}\n")
        f.write(f"  - by near-degeneracy  : {sum(1 for c in crossings if c['type']=='degeneracy')}\n")
        f.write(f"Weyl/nodal phonon pts   : {n_weyl}  (Berry phase ≈ π)\n")
        f.write("\nNote: Berry phase = π indicates a topologically non-trivial crossing.\n")
        f.write("      Verify with a full Chern number calculation on a closed surface.\n")
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Topological phonon analysis: detect Weyl/Dirac phonon points with GRACE."
    )
    parser.add_argument("--poscar",  default="POSCAR",       help="Unit cell POSCAR.")
    parser.add_argument("--model",   default="GRACE-2L-OAM", help="GRACE model.")
    parser.add_argument("--path",    default="GXMGZ",        help="k-path string (ASE notation, default: GXMGZ).")
    parser.add_argument("--nkpts",   type=int, default=200,  help="Number of k-points (default: 200).")
    parser.add_argument("--supercell", type=int, nargs=3, default=[3,3,3],
                        metavar=("Nx","Ny","Nz"), help="Phonon supercell (default: 3 3 3).")
    parser.add_argument("--delta",   type=float, default=0.015,
                        help="Displacement amplitude in Å (default: 0.015).")
    parser.add_argument("--tol",     type=float, default=0.5,
                        help="Crossing detection threshold in THz (default: 0.5).")
    parser.add_argument("--berry",   action="store_true",
                        help="Compute Berry phase at each detected crossing (slower).")
    parser.add_argument("--loop_pts",type=int, default=30,
                        help="k-points on Berry phase loop (default: 30).")
    parser.add_argument("--loop_r",  type=float, default=0.02,
                        help="Berry loop radius in reduced coords (default: 0.02).")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)

    atoms = read(args.poscar, format="vasp")
    calc  = load_model(args.model)

    ph = build_phonon_object(atoms, calc, tuple(args.supercell), args.delta)

    q_pts, omega_kn, u_kn, x_coords, special_x = get_bands_and_modes(
        ph, atoms, args.path, args.nkpts
    )

    n_imaginary = int(np.sum(omega_kn < 0))
    n_modes = omega_kn.shape[1]
    print(f"Modes: {n_modes}  |  Imaginary: {n_imaginary}")

    crossings = detect_crossings(omega_kn, x_coords, tol_THz=args.tol)
    print(f"Detected {len(crossings)} crossing(s) with threshold {args.tol} THz")

    # Optional Berry phase computation
    if args.berry and u_kn is not None:
        print(f"\nComputing Berry phase at each crossing (loop_pts={args.loop_pts}, r={args.loop_r})...")
        for i, c in enumerate(crossings):
            q0 = q_pts[c["k_idx"]]
            gamma = compute_berry_phase_at_crossing(
                ph, atoms, q0, c["band_lo"],
                n_loop=args.loop_pts, radius=args.loop_r
            )
            c["berry"] = gamma
            cls = classify_crossing(gamma)
            print(f"  Crossing {i}: bands {c['band_lo']}-{c['band_hi']}  "
                  f"γ = {np.degrees(gamma):.1f}°  → {cls}" if gamma is not None
                  else f"  Crossing {i}: Berry phase failed")

    plot_bands_with_crossings(x_coords, omega_kn, crossings, special_x)
    write_crossings(crossings, q_pts)
    write_summary(crossings, n_modes, n_imaginary)


if __name__ == "__main__":
    main()
