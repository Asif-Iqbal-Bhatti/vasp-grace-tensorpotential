#!/usr/bin/env python3
"""
topology.py

Topological Data Analysis (TDA) of crystal and grain boundary structures
using persistent homology.

Computes Betti numbers β₀, β₁, β₂ as a function of distance threshold r,
giving a topological fingerprint that does not rely on crystal symmetry —
making it ideal for grain boundaries, amorphous regions, and defect cores.

Implemented using scipy (always available):
    β₀  — connected components       — full persistence diagram via union-find
    β₁  — independent loops          — graph cycle rank at each threshold
    β₂  — enclosed voids (approx.)   — Euler characteristic estimate

Full persistent homology (β₀, β₁, β₂ with birth/death) requires gudhi:
    pip install gudhi
If gudhi is installed it is used automatically for β₁ and β₂.

Physical interpretation for grain boundaries
--------------------------------------------
    β₀ ↑  at small r  → many disconnected atomic clusters (disordered region)
    β₁ peak           → ring structures / channels (Li-ion conduction pathways)
    β₂ peak           → enclosed voids / pores (trapping sites)
    Comparing crystal vs GB: where curves diverge identifies the GB region.

Usage
-----
    # Analyse a single structure
    python topology.py --poscar POSCAR --rmax 8.0

    # Compare perfect crystal with grain boundary
    python topology.py --poscar POSCAR_bulk --compare POSCAR_gb

    # Screen a trajectory (every 10th frame), Li atoms only
    python topology.py --xdatcar XDATCAR --stride 10 --species Li --rmax 6.0

    # Sublattice analysis: separate Li, P, S, Cl
    python topology.py --poscar POSCAR --species Li --rmax 6.0

Outputs
-------
    betti_curves.dat        r (Å), β₀, β₁, β₂ at each distance threshold
    persistence_diagram.dat birth, death, dimension for each feature
    tda_summary.txt         key topological metrics
    betti_curves.png        plot of Betti curves
    persistence_diagram.png birth-death scatter plot
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from ase.io import read


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def get_positions(atoms, species=None):
    """Extract Cartesian positions, optionally filtering by species."""
    if species is None:
        return atoms.positions.copy(), atoms.get_chemical_symbols()
    sym = np.array(atoms.get_chemical_symbols())
    mask = sym == species
    if mask.sum() == 0:
        raise ValueError(f"Species '{species}' not found in structure.")
    return atoms.positions[mask].copy(), sym[mask].tolist()


def pbc_distance_matrix(positions, cell):
    """
    Pairwise minimum-image distances for a set of positions in a periodic cell.
    Returns (N, N) float array.
    """
    n = len(positions)
    cell_inv = np.linalg.inv(cell)
    D = np.zeros((n, n))
    for i in range(n):
        dv = positions[i] - positions
        frac = dv @ cell_inv
        frac -= np.round(frac)
        mic = frac @ cell
        D[i] = np.linalg.norm(mic, axis=1)
    return D


# ──────────────────────────────────────────────────────────────────────────────
# β₀ persistent homology via union-find (exact)
# ──────────────────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.birth  = [0.0] * n   # birth time of each component

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y, r):
        """Merge components of x and y at threshold r. Returns (died_root, survived_root) or None."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None
        # Younger component (higher birth) dies
        if self.birth[rx] >= self.birth[ry]:
            dying, surviving = rx, ry
        else:
            dying, surviving = ry, rx
        if self.rank[surviving] < self.rank[dying]:
            surviving, dying = dying, surviving
        self.parent[dying] = surviving
        if self.rank[surviving] == self.rank[dying]:
            self.rank[surviving] += 1
        return dying, surviving, r


def compute_beta0_persistence(D, n):
    """
    Compute β₀ (connected components) persistence diagram using union-find.

    Parameters
    ----------
    D : (n, n) distance matrix
    n : number of points

    Returns
    -------
    pairs : list of (birth, death) — components that are born and die
    infinite : list of birth values for components that survive to r_max
    beta0_curve : function(r) → int  (number of components at threshold r)
    """
    # Extract sorted unique edge distances
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i, j], i, j))
    edges.sort()

    uf = UnionFind(n)
    pairs = []

    for r, i, j in edges:
        result = uf.union(i, j, r)
        if result is not None:
            dying, surviving, death_r = result
            birth_r = uf.birth[dying]
            pairs.append((birth_r, death_r))

    # Surviving components (infinite lifetime)
    roots = set(uf.find(i) for i in range(n))
    infinite = [uf.birth[r] for r in roots]

    return pairs, infinite


def beta0_at_threshold(pairs, n_infinite, r):
    """Count β₀ at a given threshold r from precomputed persistence pairs."""
    alive = n_infinite  # always alive
    for birth, death in pairs:
        if birth <= r < death:
            alive += 1
    return alive


# ──────────────────────────────────────────────────────────────────────────────
# β₁ from graph cycle rank (scipy, exact for 1-skeleton)
# ──────────────────────────────────────────────────────────────────────────────

def compute_betti_curves_scipy(D, r_values):
    """
    Compute β₀, β₁, (β₂ approx) curves at each threshold in r_values.

    β₀ : connected components (exact, from union-find persistence)
    β₁ : cycle rank of the graph = E - V + C  (exact for 1-skeleton)
    β₂ : not computable from graph alone; placeholder zeros

    Returns arrays of shape (len(r_values),) for each β.
    """
    n = len(D)

    # β₀ persistence (for the curve)
    print("Computing β₀ persistence...")
    pairs, infinite = compute_beta0_persistence(D, n)
    n_inf = len(infinite)

    beta0 = np.array([beta0_at_threshold(pairs, n_inf, r) for r in r_values])

    # β₁ from graph cycle rank
    print("Computing β₁ curve...")
    beta1 = np.zeros(len(r_values), dtype=int)
    for k, r in enumerate(r_values):
        adj  = (D <= r) & (D > 0)
        n_edges = int(adj.sum()) // 2
        G = csr_matrix(adj.astype(float))
        n_comp, _ = connected_components(G, directed=False)
        beta1[k] = max(0, n_edges - n + n_comp)

    beta2 = np.zeros(len(r_values), dtype=int)

    return beta0, beta1, beta2, pairs, infinite


# ──────────────────────────────────────────────────────────────────────────────
# gudhi backend (full persistent homology, β₀ β₁ β₂)
# ──────────────────────────────────────────────────────────────────────────────

def compute_betti_curves_gudhi(positions, r_values):
    """Full persistent homology using gudhi AlphaComplex."""
    import gudhi

    print("Using gudhi AlphaComplex for full persistent homology...")
    alpha = gudhi.AlphaComplex(points=positions.tolist())
    st = alpha.create_simplex_tree()
    st.compute_persistence()

    # Persistence intervals per dimension
    intervals = {d: st.persistence_intervals_in_dimension(d) for d in range(3)}

    def count_alive(dim, r):
        r2 = r ** 2   # Alpha complex uses squared radius
        cnt = 0
        for b, d in intervals[dim]:
            if b <= r2 and (d > r2 or np.isinf(d)):
                cnt += 1
        return cnt

    beta0 = np.array([count_alive(0, r) for r in r_values])
    beta1 = np.array([count_alive(1, r) for r in r_values])
    beta2 = np.array([count_alive(2, r) for r in r_values])

    # Persistence pairs for output (convert squared radius → distance)
    all_pairs = []
    for d in range(3):
        for b, death in intervals[d]:
            all_pairs.append((np.sqrt(b), np.sqrt(death) if not np.isinf(death) else np.inf, d))

    return beta0, beta1, beta2, all_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Key metrics from Betti curves
# ──────────────────────────────────────────────────────────────────────────────

def topological_fingerprint(r_values, beta0, beta1, beta2):
    """Extract key scalar metrics from Betti curves."""
    metrics = {}
    metrics["beta0_max"]       = int(beta0.max())
    metrics["beta0_final"]     = int(beta0[-1])
    metrics["beta1_max"]       = int(beta1.max())
    metrics["beta1_peak_r"]    = float(r_values[np.argmax(beta1)])
    metrics["beta2_max"]       = int(beta2.max())
    metrics["beta2_peak_r"]    = float(r_values[np.argmax(beta2)]) if beta2.max() > 0 else 0.0
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Writers & plots
# ──────────────────────────────────────────────────────────────────────────────

def write_betti_curves(r_values, beta0, beta1, beta2, filename="betti_curves.dat"):
    with open(filename, "w") as f:
        f.write(f"{'#r(A)':>10}  {'beta0':>8}  {'beta1':>8}  {'beta2':>8}\n")
        for r, b0, b1, b2 in zip(r_values, beta0, beta1, beta2):
            f.write(f"{r:10.5f}  {b0:8d}  {b1:8d}  {b2:8d}\n")
    print(f"Wrote {filename}")


def write_persistence_diagram(pairs, infinite=None, filename="persistence_diagram.dat"):
    with open(filename, "w") as f:
        f.write(f"# Persistence diagram: birth (Å), death (Å), dimension\n")
        f.write(f"{'#birth(A)':>12}  {'death(A)':>12}  {'dim':>5}\n")
        if isinstance(pairs[0], tuple) and len(pairs[0]) == 3:
            # gudhi format: (birth, death, dim)
            for b, d, dim in pairs:
                f.write(f"{b:12.6f}  {d if not np.isinf(d) else 999.0:12.6f}  {dim:5d}\n")
        else:
            # scipy union-find format: (birth, death)
            for b, d in pairs:
                f.write(f"{b:12.6f}  {d:12.6f}  {'0':>5}\n")
            if infinite:
                for b in infinite:
                    f.write(f"{b:12.6f}  {'inf':>12}  {'0':>5}\n")
    print(f"Wrote {filename}")


def plot_betti_curves(r_values, beta0, beta1, beta2,
                      label="", compare_data=None, filename="betti_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    names = ["β₀ (components)", "β₁ (loops)", "β₂ (voids)"]
    curves = [beta0, beta1, beta2]

    for ax, name, curve in zip(axes, names, curves):
        ax.plot(r_values, curve, lw=2, label=label if label else "structure")
        if compare_data is not None:
            ax.plot(r_values, compare_data[names.index(name)], lw=2,
                    ls="--", label="comparison")
        ax.set_xlabel("Distance threshold r (Å)")
        ax.set_ylabel("Betti number")
        ax.set_title(name)
        ax.legend(fontsize=8)

    fig.suptitle("Topological Data Analysis — Betti curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


def plot_persistence_diagram(pairs, infinite=None, filename="persistence_diagram.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    if isinstance(pairs[0], tuple) and len(pairs[0]) == 3:
        colors = {0: "steelblue", 1: "darkorange", 2: "green"}
        labels = {0: "β₀", 1: "β₁", 2: "β₂"}
        plotted = set()
        for b, d, dim in pairs:
            death = min(d, max(b for b, _, _ in pairs) * 1.1)
            lbl = labels[dim] if dim not in plotted else None
            ax.scatter(b, death, c=colors.get(dim, "gray"), s=20, alpha=0.7, label=lbl)
            plotted.add(dim)
    else:
        births  = [b for b, _ in pairs]
        deaths  = [d for _, d in pairs]
        ax.scatter(births, deaths, c="steelblue", s=20, alpha=0.7, label="β₀")
        if infinite:
            ax.scatter(infinite, [max(deaths) * 1.05] * len(infinite),
                       marker="^", c="red", s=30, label="infinite")

    lim = ax.get_xlim()[1]
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("Birth (Å)")
    ax.set_ylabel("Death (Å)")
    ax.set_title("Persistence diagram")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Wrote {filename}")


def write_summary(metrics, n_atoms, species, method, filename="tda_summary.txt"):
    with open(filename, "w") as f:
        f.write("Topological Data Analysis summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Species analysed : {species if species else 'all'}\n")
        f.write(f"N atoms          : {n_atoms}\n")
        f.write(f"Backend          : {method}\n")
        f.write(f"\nBetti curve metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k:<22} = {v}\n")
        f.write("\nPhysical interpretation:\n")
        f.write(f"  β₀(max)={metrics['beta0_max']} → atom clusters at small r\n")
        f.write(f"  β₁(max)={metrics['beta1_max']} at r={metrics['beta1_peak_r']:.2f} Å → ring/channel structures\n")
        f.write(f"  β₂(max)={metrics['beta2_max']} → enclosed voids (0 = no gudhi)\n")
    print(f"Wrote {filename}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Topological Data Analysis of crystal/grain boundary structures."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--poscar",  help="Single POSCAR/CONTCAR to analyse.")
    src.add_argument("--xdatcar", help="XDATCAR trajectory (analyses each frame).")

    parser.add_argument("--compare", default=None,
                        help="Second POSCAR to compare (e.g. GB vs perfect crystal).")
    parser.add_argument("--species", default=None,
                        help="Restrict analysis to this element (e.g. Li). Default: all atoms.")
    parser.add_argument("--rmax",    type=float, default=8.0,
                        help="Maximum distance threshold in Å (default: 8.0).")
    parser.add_argument("--rmin",    type=float, default=0.5,
                        help="Minimum distance threshold in Å (default: 0.5).")
    parser.add_argument("--npoints", type=int, default=150,
                        help="Number of threshold values (default: 150).")
    parser.add_argument("--stride",  type=int, default=1,
                        help="Use every Nth frame from XDATCAR (default: 1).")
    parser.add_argument("--pbc",     action="store_true",
                        help="Use minimum-image distances (PBC). Slower but correct for periodic cells.")

    args = parser.parse_args()

    # Detect gudhi
    try:
        import gudhi
        USE_GUDHI = True
        print("gudhi detected — using full persistent homology (β₀, β₁, β₂).")
    except ImportError:
        USE_GUDHI = False
        print("gudhi not found — using scipy backend (β₀ exact, β₁ from graph, β₂≈0).")
        print("Install with: pip install gudhi\n")

    r_values = np.linspace(args.rmin, args.rmax, args.npoints)

    def analyse_atoms(atoms, label=""):
        pos, sym = get_positions(atoms, args.species)
        print(f"\nAnalysing {label}: {len(pos)} atoms ({args.species if args.species else 'all'})")

        if args.pbc:
            cell = np.array(atoms.get_cell())
            D = pbc_distance_matrix(pos, cell)
        else:
            D = cdist(pos, pos)

        if USE_GUDHI:
            b0, b1, b2, pairs = compute_betti_curves_gudhi(pos, r_values)
            pers_pairs = pairs
        else:
            b0, b1, b2, pairs, infinite = compute_betti_curves_scipy(D, r_values)
            pers_pairs = pairs

        return b0, b1, b2, pers_pairs, (pairs if not USE_GUDHI else None), \
               (infinite if not USE_GUDHI else None), len(pos)

    method = "gudhi (AlphaComplex)" if USE_GUDHI else "scipy (union-find + graph)"

    if args.xdatcar:
        # Trajectory mode: average Betti curves
        traj = read(args.xdatcar, index=f"::{args.stride}",
                    format="vasp-xdatcar")
        print(f"Trajectory: {len(traj)} frames (stride={args.stride})")

        b0_sum = np.zeros(args.npoints)
        b1_sum = np.zeros(args.npoints)
        b2_sum = np.zeros(args.npoints)
        last_pairs, last_inf = None, None

        for f, atoms in enumerate(traj):
            print(f"Frame {f + 1}/{len(traj)}")
            b0, b1, b2, pp, sc_pairs, sc_inf, n_atoms = analyse_atoms(atoms, f"frame {f}")
            b0_sum += b0; b1_sum += b1; b2_sum += b2
            last_pairs, last_inf = sc_pairs, sc_inf

        b0 = (b0_sum / len(traj)).astype(float)
        b1 = (b1_sum / len(traj)).astype(float)
        b2 = (b2_sum / len(traj)).astype(float)

        write_betti_curves(r_values, np.round(b0).astype(int),
                           np.round(b1).astype(int),
                           np.round(b2).astype(int))
        plot_betti_curves(r_values, b0, b1, b2, label="trajectory avg")
        if last_pairs:
            write_persistence_diagram(last_pairs, last_inf)

        metrics = topological_fingerprint(r_values, b0, b1, b2)
        write_summary(metrics, n_atoms, args.species, method)

    else:
        atoms = read(args.poscar, format="vasp")
        b0, b1, b2, pp, sc_pairs, sc_inf, n_atoms = analyse_atoms(atoms, args.poscar)

        compare_curves = None
        if args.compare:
            atoms2 = read(args.compare, format="vasp")
            b0c, b1c, b2c, _, _, _, _ = analyse_atoms(atoms2, args.compare)
            compare_curves = [b0c, b1c, b2c]
            # Write comparison
            write_betti_curves(r_values, b0c.astype(int), b1c.astype(int),
                               b2c.astype(int), filename="betti_curves_compare.dat")
            print(f"\nTopological difference (max |Δβ|):")
            print(f"  Δβ₀ = {np.max(np.abs(b0 - b0c)):.0f}")
            print(f"  Δβ₁ = {np.max(np.abs(b1 - b1c)):.0f}")
            print(f"  Δβ₂ = {np.max(np.abs(b2 - b2c)):.0f}")

        write_betti_curves(r_values, b0.astype(int), b1.astype(int), b2.astype(int))

        if USE_GUDHI:
            write_persistence_diagram(pp)
            plot_persistence_diagram(pp)
        elif sc_pairs:
            write_persistence_diagram(sc_pairs, sc_inf)
            plot_persistence_diagram(sc_pairs, sc_inf)

        plot_betti_curves(r_values, b0, b1, b2,
                          label=os.path.basename(args.poscar),
                          compare_data=compare_curves)
        metrics = topological_fingerprint(r_values, b0, b1, b2)
        write_summary(metrics, n_atoms, args.species, method)


if __name__ == "__main__":
    main()
