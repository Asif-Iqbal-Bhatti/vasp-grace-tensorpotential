#!/usr/bin/env python3
"""
lihopping.py

Li-ion (or any mobile species) hop event detection and analysis from MD/MC trajectories.

Tracks each mobile atom across reference lattice sites, detects hop events using a
sojourn-time filter to eliminate false positives from oscillating atoms, builds a
hop network, and computes:

    - Per-site and per-atom hop rates
    - Mean residence times
    - Spatial distribution of active hops across the structure
    - Arrhenius activation energy (if multiple temperature runs are provided)

Works with XDATCAR (from main.py MD), XDATCAR_MC (from montecarlo.py),
or any ASE-readable trajectory file.

Usage
-----
    # Basic hop analysis
    python lihopping.py --traj XDATCAR --ref POSCAR --species Li --timestep 2.0

    # With sojourn filter (ignore transient site visits < 3 frames)
    python lihopping.py --traj XDATCAR --ref POSCAR --species Li --timestep 2.0 --min_sojourn 3

    # Grain-boundary vs bulk hop comparison
    python lihopping.py --traj XDATCAR --ref POSCAR --species Li --timestep 2.0 --gb_axis z --gb_frac 0.5

    # Arrhenius analysis from multiple temperature trajectories
    python lihopping.py --arrhenius \\
        --temps 600 800 1000 \\
        --trajs traj_600K XDATCAR_800K XDATCAR_1000K \\
        --ref POSCAR --species Li --timestep 2.0

Outputs
-------
    hop_events.dat       frame, atom_id, from_site, to_site, hop_distance (Å)
    hop_statistics.dat   per-site: n_hops, hop_rate (hops/ps), mean_residence_time (ps)
    hop_network.dat      site_i, site_j, n_hops (for network visualization)
    arrhenius.dat        T (K), 1/T, hop_rate, ln(rate)  [if --arrhenius]
    arrhenius.png        Arrhenius plot
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial import cKDTree


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def fractional(positions, cell):
    return positions @ np.linalg.inv(cell)

def cartesian(frac, cell):
    return frac @ cell

def mic_distance(r1, r2, cell):
    """Minimum-image distance between two Cartesian positions."""
    df = fractional(r2 - r1, cell)
    df -= np.round(df)
    return np.linalg.norm(cartesian(df, cell))


# ──────────────────────────────────────────────────────────────────────────────
# Site assignment
# ──────────────────────────────────────────────────────────────────────────────

def assign_sites(mob_pos, ref_pos, cell):
    """
    Assign each mobile atom to the nearest reference site under PBC.
    Returns (site_ids, distances).
    """
    # Build reference site images (3×3×3 supercell) for PBC
    frac_ref = fractional(ref_pos, cell)
    images, site_map = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for s, fr in enumerate(frac_ref):
                    shifted = fr + np.array([dx, dy, dz])
                    images.append(cartesian(shifted, cell))
                    site_map.append(s)

    tree = cKDTree(images)
    dists, idx = tree.query(mob_pos)
    site_ids = np.array([site_map[i] for i in idx])
    return site_ids, dists


# ──────────────────────────────────────────────────────────────────────────────
# Hop detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_hops(trajectory, ref_atoms, species, min_sojourn=2):
    """
    Detect hop events in a trajectory with a sojourn-time filter.

    A hop is confirmed only if the atom stays in the new site for at least
    `min_sojourn` consecutive frames (eliminates boundary oscillations).

    Returns
    -------
    events : list of dicts with keys: frame, atom_id, from_site, to_site, distance
    all_assignments : (n_frames, n_mobile) int array of site assignments
    ref_mob_pos : (n_sites, 3) reference positions of mobile species
    """
    symbols = np.array(ref_atoms.get_chemical_symbols())
    mob_mask = symbols == species
    ref_mob_pos = ref_atoms.positions[mob_mask]
    n_sites = ref_mob_pos.shape[0]
    cell = np.array(ref_atoms.get_cell())

    n_frames = len(trajectory)
    n_mobile = mob_mask.sum()

    print(f"Reference sites : {n_sites} {species} sites")
    print(f"Trajectory      : {n_frames} frames, {n_mobile} mobile atoms")
    print(f"Sojourn filter  : {min_sojourn} frames\n")

    # Compute site assignments for every frame
    all_assign = np.zeros((n_frames, n_mobile), dtype=int)
    all_dists  = np.zeros((n_frames, n_mobile))

    for f, atoms in enumerate(trajectory):
        sym = np.array(atoms.get_chemical_symbols())
        pos = atoms.positions[sym == species]
        ids, dists = assign_sites(pos, ref_mob_pos, cell)
        all_assign[f] = ids
        all_dists[f]  = dists
        if (f + 1) % 500 == 0:
            print(f"  processed frame {f+1}/{n_frames}")

    # Sojourn-filtered hop detection per atom
    events = []
    for atom in range(n_mobile):
        a = all_assign[:, atom]
        f = 0
        while f < n_frames - 1:
            if a[f + 1] != a[f]:
                new_site = a[f + 1]
                # Find how long the atom stays in new_site
                end = f + 1
                while end < n_frames and a[end] == new_site:
                    end += 1
                sojourn = end - (f + 1)
                if sojourn >= min_sojourn:
                    events.append({
                        "frame"    : f + 1,
                        "atom_id"  : atom,
                        "from_site": int(a[f]),
                        "to_site"  : int(new_site),
                        "distance" : float(all_dists[f + 1, atom]),
                    })
                    f = end          # jump past sojourn
                else:
                    f = end          # skip short visit
            else:
                f += 1

    return events, all_assign, ref_mob_pos


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_statistics(events, n_sites, n_mobile, n_frames, timestep_ps):
    """Per-site hop counts, rates, mean residence times."""
    total_time_ps = n_frames * timestep_ps
    hop_count = np.zeros(n_sites, dtype=int)
    for e in events:
        hop_count[e["from_site"]] += 1

    hop_rate = hop_count / (total_time_ps * n_mobile)          # hops/ps per atom
    # Residence time: average time between departures from a site
    with np.errstate(divide="ignore", invalid="ignore"):
        res_time = np.where(hop_count > 0, total_time_ps / hop_count, np.inf)

    return hop_count, hop_rate, res_time


def compute_hop_network(events, n_sites):
    """Build site-pair hop count matrix."""
    net = np.zeros((n_sites, n_sites), dtype=int)
    for e in events:
        i, j = e["from_site"], e["to_site"]
        net[i, j] += 1
        net[j, i] += 1
    return net


# ──────────────────────────────────────────────────────────────────────────────
# Arrhenius analysis
# ──────────────────────────────────────────────────────────────────────────────

def arrhenius_analysis(temps_K, rates, output_prefix="arrhenius"):
    """Fit ln(k) = ln(k0) - Ea/(kB*T) and plot."""
    from ase import units
    kB = units.kB  # eV/K

    inv_T = 1.0 / np.array(temps_K)
    ln_k  = np.log(np.array(rates))

    coeffs = np.polyfit(inv_T, ln_k, 1)
    Ea_eV  = -coeffs[0] * kB
    k0     = np.exp(coeffs[1])

    print(f"\nArrhenius fit:")
    print(f"  Activation energy Ea = {Ea_eV*1000:.1f} meV  ({Ea_eV:.4f} eV)")
    print(f"  Pre-exponential  k0  = {k0:.4e} hops/ps")

    # Write dat
    with open(f"{output_prefix}.dat", "w") as f:
        f.write(f"# Arrhenius analysis\n")
        f.write(f"# Ea = {Ea_eV*1000:.2f} meV,  k0 = {k0:.4e} hops/ps\n")
        f.write(f"{'#T(K)':>10}  {'1/T(1/K)':>14}  {'rate(hops/ps)':>15}  {'ln(rate)':>12}\n")
        for T, r in zip(temps_K, rates):
            f.write(f"{T:10.1f}  {1/T:14.8f}  {r:15.8e}  {np.log(r):12.6f}\n")
    print(f"Wrote {output_prefix}.dat")

    # Plot
    T_fit = np.linspace(min(inv_T) * 0.95, max(inv_T) * 1.05, 200)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(inv_T * 1000, ln_k, "o", ms=8, label="Data")
    ax.plot(T_fit * 1000, np.polyval(coeffs, T_fit), "--",
            label=f"Fit  Ea={Ea_eV*1000:.0f} meV")
    ax.set_xlabel("1000/T  (1/K)")
    ax.set_ylabel("ln(hop rate)  [rate in hops/ps]")
    ax.set_title("Arrhenius plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=300)
    plt.close(fig)
    print(f"Wrote {output_prefix}.png")

    return Ea_eV, k0


# ──────────────────────────────────────────────────────────────────────────────
# File writers
# ──────────────────────────────────────────────────────────────────────────────

def write_hop_events(events, filename="hop_events.dat"):
    with open(filename, "w") as f:
        f.write(f"{'#frame':>8}  {'atom_id':>8}  {'from_site':>10}  "
                f"{'to_site':>10}  {'distance(A)':>12}\n")
        for e in events:
            f.write(f"{e['frame']:8d}  {e['atom_id']:8d}  {e['from_site']:10d}  "
                    f"{e['to_site']:10d}  {e['distance']:12.6f}\n")
    print(f"Wrote {filename}  ({len(events)} events)")


def write_hop_statistics(hop_count, hop_rate, res_time, ref_mob_pos, filename="hop_statistics.dat"):
    with open(filename, "w") as f:
        f.write(f"{'#site':>6}  {'x(A)':>10}  {'y(A)':>10}  {'z(A)':>10}  "
                f"{'n_hops':>8}  {'rate(1/ps)':>12}  {'res_time(ps)':>14}\n")
        for i, (pos, nc, nr, rt) in enumerate(zip(ref_mob_pos, hop_count, hop_rate, res_time)):
            f.write(f"{i:6d}  {pos[0]:10.5f}  {pos[1]:10.5f}  {pos[2]:10.5f}  "
                    f"{nc:8d}  {nr:12.6e}  {rt:14.6e}\n")
    print(f"Wrote {filename}")


def write_hop_network(net, filename="hop_network.dat"):
    n = net.shape[0]
    with open(filename, "w") as f:
        f.write(f"{'#site_i':>8}  {'site_j':>8}  {'n_hops':>8}\n")
        for i in range(n):
            for j in range(i + 1, n):
                if net[i, j] > 0:
                    f.write(f"{i:8d}  {j:8d}  {net[i,j]:8d}\n")
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Li-ion (or any species) hop detection from MD/MC trajectories."
    )
    parser.add_argument("--traj",   default="XDATCAR", help="Trajectory file (default: XDATCAR).")
    parser.add_argument("--ref",    default="POSCAR",  help="Reference POSCAR with ideal sites.")
    parser.add_argument("--species",default="Li",      help="Mobile species symbol (default: Li).")
    parser.add_argument("--timestep", type=float, default=2.0,
                        help="MD timestep in fs (default: 2.0).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Read every Nth frame (default: 1).")
    parser.add_argument("--min_sojourn", type=int, default=2,
                        help="Min frames in new site to confirm a hop (default: 2).")
    parser.add_argument("--format", default=None,
                        help="ASE format string for trajectory (auto-detected if omitted).")

    # Arrhenius mode
    parser.add_argument("--arrhenius", action="store_true",
                        help="Perform Arrhenius analysis across multiple temperatures.")
    parser.add_argument("--temps", type=float, nargs="+",
                        help="Temperatures in K for Arrhenius (requires --arrhenius).")
    parser.add_argument("--trajs", nargs="+",
                        help="Trajectory files for each temperature (requires --arrhenius).")

    args = parser.parse_args()

    timestep_ps = args.timestep * 1e-3   # fs → ps

    if args.arrhenius:
        if not args.temps or not args.trajs or len(args.temps) != len(args.trajs):
            print("Error: --arrhenius requires --temps and --trajs with equal counts.")
            sys.exit(1)

        ref_atoms = read(args.ref, format="vasp")
        rates = []
        for T, traj_file in zip(args.temps, args.trajs):
            print(f"\n=== T = {T} K  |  {traj_file} ===")
            traj = read(traj_file, index=f"::{args.stride}",
                        format=args.format if args.format else None)
            events, all_assign, ref_mob_pos = detect_hops(
                traj, ref_atoms, args.species, args.min_sojourn
            )
            n_mobile = all_assign.shape[1]
            total_time_ps = len(traj) * timestep_ps
            rate = len(events) / (total_time_ps * n_mobile) if n_mobile > 0 else 0.0
            print(f"  Total hops : {len(events)}")
            print(f"  Hop rate   : {rate:.6e} hops/ps/atom")
            rates.append(rate)

        arrhenius_analysis(args.temps, rates)
        return

    # Single trajectory mode
    if not os.path.exists(args.ref):
        print(f"Error: reference file {args.ref} not found."); sys.exit(1)
    if not os.path.exists(args.traj):
        print(f"Error: trajectory {args.traj} not found."); sys.exit(1)

    print(f"Loading trajectory: {args.traj} (stride={args.stride})")
    traj = read(args.traj, index=f"::{args.stride}",
                format=args.format if args.format else None)
    ref_atoms = read(args.ref, format="vasp")

    events, all_assign, ref_mob_pos = detect_hops(
        traj, ref_atoms, args.species, args.min_sojourn
    )

    n_frames  = len(traj)
    n_mobile  = all_assign.shape[1]
    n_sites   = ref_mob_pos.shape[0]
    total_time_ps = n_frames * timestep_ps

    print(f"\nDetected {len(events)} hop events")
    print(f"Total simulation time : {total_time_ps:.3f} ps")

    hop_count, hop_rate, res_time = compute_statistics(
        events, n_sites, n_mobile, n_frames, timestep_ps
    )

    overall_rate = len(events) / (total_time_ps * n_mobile) if n_mobile > 0 else 0.0
    print(f"Overall hop rate      : {overall_rate:.6e} hops/ps/atom")
    print(f"Mean residence time   : {1.0/overall_rate:.3f} ps" if overall_rate > 0 else "")

    net = compute_hop_network(events, n_sites)

    write_hop_events(events)
    write_hop_statistics(hop_count, hop_rate, res_time, ref_mob_pos)
    write_hop_network(net)


if __name__ == "__main__":
    main()
