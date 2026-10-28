#!/usr/bin/env python3
"""
symmetry_stability.py

Does a crystal's band-insulator verdict survive relaxation, soft phonons and
temperature? Combines the filling rule implemented in symlint with GRACE dynamics.

Physics background
------------------
Watanabe, Po, Zaletel and Vishwanath (PRL 117, 096404, 2016) proved that a
nonmagnetic crystal with space group G and ν electrons per primitive cell can be
a band insulator only if ν belongs to an allowed set S_G. Otherwise it is forced
to be gapless: a metal, or a semimetal with symmetry-enforced band touchings.

symlint applies the rule to a fixed structure. The verdict is only as reliable
as the space group, and the space group can change:

    relaxation     the input may not sit at the symmetric minimum
    soft phonons   an imaginary mode at Γ lowers G at fixed ν; one at the zone
                   boundary also multiplies the cell and changes ν, the
                   Peierls / charge-density-wave route to opening a gap
    temperature    the time-averaged structure (what diffraction sees) may
                   differ from the 0 K one

This module relaxes the structure with GRACE (symmetry constrained), computes
phonons at Γ and at the zone-boundary points commensurate with an n×n×n
supercell, follows every soft mode into its distorted structure and applies
the rule again. Optional NVT MD checks the time-averaged structure. It also
reports structural chirality and, for UQ-capable GRACE models, the CALM
extrapolation grade γ, so you can see where the potential is trustworthy.

Usage
-----
    # Relax, phonons at Γ and zone boundary, verdict at each stage
    grace-symmetry --poscar POSCAR

    # Add 300 K molecular dynamics and CALM uncertainty
    grace-symmetry --poscar POSCAR --model GRACE-2L-OMAT-medium-ft-AM --md 300

    # Also write VASP + SOC inputs and an IrRep script for the DFT check
    grace-symmetry --poscar POSCAR --handoff

Outputs
-------
    symmetry_report.txt   summary of every stage and the overall assessment
    symmetry_report.json  all numbers
    soft_modes.dat        q, frequency, degeneracy, distorted space group, verdict
    POSCAR_primitive      relaxed, standardized primitive cell
    dft_handoff/          VASP + SOC inputs and IrRep script (with --handoff)

The verdict needs symlint (pip install -e ".[condmat]"). Chirality, phonons and
MD run without it.
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
import importlib.util

import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import spglib
from ase import Atoms, units
from ase.io import read, write

try:
    from ..calculator import load_calculator, enable_calm, DEFAULT_MODEL
    from .chirality import chirality_report
except ImportError:  # run as a plain script; needs vasp_grace installed
    from vasp_grace.calculator import load_calculator, enable_calm, DEFAULT_MODEL
    from vasp_grace.condmat.chirality import chirality_report


THZ_PER_EV = units._e / units._hplanck / 1e12
PROTECTED = ("odd_filling", "symmetry_enforced")


# ──────────────────────────────────────────────────────────────────────────────
# Symmetry and verdict
# ──────────────────────────────────────────────────────────────────────────────

def _cell(atoms):
    return (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)


def space_group(atoms, symprec=1e-3):
    """(number, symbol, order of the point group part) or (None, None, 0)."""
    ds = spglib.get_symmetry_dataset(_cell(atoms), symprec=symprec)
    if ds is None:
        return None, None, 0
    return int(ds.number), str(ds.international), len(ds.rotations)


def standard_primitive(atoms, symprec=1e-3):
    """Standardized primitive cell from spglib, or a copy if spglib fails."""
    std = spglib.standardize_cell(_cell(atoms), to_primitive=True, symprec=symprec)
    if std is None:
        return atoms.copy()
    lattice, scaled, numbers = std
    return Atoms(numbers=numbers, cell=lattice, scaled_positions=scaled, pbc=True)


def filling_verdict(atoms, symprec=1e-3, mode="soc"):
    """symlint verdict for `atoms`, or None when symlint is not installed."""
    try:
        from symlint.check import check_structure
    except ImportError:
        return None
    r = check_structure(atoms, symprec=symprec, mode=mode)
    keep = ("sg_number", "sg_symbol", "nu", "allowed_rule", "verdict",
            "can_be_insulator", "symprec_sensitive", "verdict_sensitive", "explanation")
    return {k: r.get(k) for k in keep}


# ──────────────────────────────────────────────────────────────────────────────
# GRACE: relaxation, CALM γ
# ──────────────────────────────────────────────────────────────────────────────

def relax_symmetric(atoms, calc, fmax=0.002, steps=300, symprec=1e-3):
    """Relax positions and cell with the space group held fixed."""
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    a = atoms.copy()
    a.calc = calc
    a.set_constraint(FixSymmetry(a, symprec=symprec))
    opt = LBFGS(FrechetCellFilter(a), logfile="relax.log")
    converged = bool(opt.run(fmax=fmax, steps=steps))
    a.set_constraint()
    return a, converged, opt.get_number_of_steps()


def calm_gamma(atoms, calc):
    """(max γ, mean γ) with CALM switched on for this one evaluation."""
    a = atoms.copy()
    a.calc = calc
    calc.enable_uq("gamma_only")
    try:
        a.get_potential_energy()
        g = np.asarray(calc.results["gamma"], dtype=float)
    finally:
        calc.disable_uq()
    return float(g.max()), float(g.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Phonons at commensurate q and soft-mode distortions
# ──────────────────────────────────────────────────────────────────────────────

def commensurate_qpoints(n):
    """q = (i, j, k)/n in reduced coordinates, one of each ±q pair."""
    seen, qs = set(), []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                key = (i % n, j % n, k % n)
                mirror = ((-i) % n, (-j) % n, (-k) % n)
                if mirror in seen:
                    continue
                seen.add(key)
                qs.append(np.array(key, dtype=float) / n)
    return np.array(qs)


def phonons_at(prim, calc, n, delta, workdir):
    """Frequencies (THz) and mass-scaled mode patterns at the commensurate q-points."""
    from ase.phonons import Phonons

    ph = Phonons(prim, calc, supercell=(n, n, n), delta=delta,
                 name=os.path.join(workdir, "phonon"))
    ph.run()
    ph.read(acoustic=True)
    qs = commensurate_qpoints(n)
    omega_kl, u_kl = ph.band_structure(qs, modes=True, verbose=False)
    return qs, omega_kl * THZ_PER_EV, u_kl


def degenerate_groups(freqs, tol=0.05, skip=()):
    """Indices of (near-)degenerate modes, lowest frequency first, leaving out `skip`."""
    skip = set(skip)
    order = [i for i in np.argsort(freqs) if i not in skip]
    if not order:
        return []
    groups, current = [], [order[0]]
    for i in order[1:]:
        if abs(freqs[i] - freqs[current[-1]]) < tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def modulated_supercell(prim, q, pattern, n, amplitude):
    """
    n×n×n supercell displaced along u(R, κ) = Re[e_κ(q) exp(2πi q·R)].

    `pattern` is ASE's mode (natoms, 3), already scaled by 1/√m. The global phase
    is fixed so the largest component is real. Max displacement = amplitude (Å).
    """
    e = np.asarray(pattern, dtype=complex)
    j = np.argmax(np.abs(e.ravel()))
    e = e * np.exp(-1j * np.angle(e.ravel()[j]))

    # same image order as Atoms.repeat
    R = np.array([(a, b, c) for a in range(n) for b in range(n) for c in range(n)])
    phase = np.exp(2j * np.pi * (R @ q))
    u = np.real(phase[:, None, None] * e[None, :, :]).reshape(-1, 3)
    umax = np.linalg.norm(u, axis=1).max()
    if umax < 1e-10:
        return None
    sc = prim.repeat((n, n, n))
    sc.positions += u * (amplitude / umax)
    return sc


def analyse_soft_modes(prim, qs, freqs, modes, n, imag_tol, amplitude, symprec, mode,
                       calc=None, calm_ok=False):
    """
    Follow every imaginary mode (|ν| > imag_tol) into its distorted supercell.

    For a degenerate set, each basis vector and their sum are tried, because a
    single numpy eigenvector picks an arbitrary direction in the subspace.
    """
    rows = []
    lowest = None
    for iq, q in enumerate(qs):
        f = freqs[iq]
        # at Γ the three smallest |ν| are the rigid translations
        acoustic = np.argsort(np.abs(f))[:3] if np.allclose(q, 0) else ()
        for g in degenerate_groups(f, skip=[int(i) for i in acoustic]):
            nu = float(np.mean(f[g]))
            if nu > -imag_tol:
                if lowest is None or nu < lowest[1]:
                    lowest = (q.tolist(), nu)
                continue
            trials = [modes[iq, i] for i in g]
            if len(g) > 1:
                trials.append(np.sum([modes[iq, i] for i in g], axis=0))
            for t_idx, pattern in enumerate(trials):
                sc = modulated_supercell(prim, q, pattern, n, amplitude)
                if sc is None:
                    continue
                sg, sym, order = space_group(sc, symprec)
                ver = filling_verdict(sc, symprec=symprec, mode=mode)
                row = {
                    "q": q.tolist(),
                    "frequency_THz": nu,
                    "degeneracy": len(g),
                    "trial": "sum" if t_idx == len(g) else f"basis {t_idx + 1}",
                    "distorted_sg_number": sg,
                    "distorted_sg_symbol": sym,
                    "distorted_order": order,
                    "nu_distorted": ver["nu"] if ver else None,
                    "verdict_distorted": ver["verdict"] if ver else None,
                    "gamma_max": None,
                }
                if calm_ok:
                    row["gamma_max"] = calm_gamma(sc, calc)[0]
                rows.append(row)
    return rows, lowest


# ──────────────────────────────────────────────────────────────────────────────
# Finite-temperature average structure
# ──────────────────────────────────────────────────────────────────────────────

def md_average_structure(prim, calc, temperature, steps, timestep, equil_frac,
                         min_length, sample_every, seed, symprec_avg, mode, calm_ok):
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    heights = 1.0 / np.linalg.norm(prim.cell.reciprocal(), axis=1)
    reps = [max(1, int(np.ceil(min_length / h))) for h in heights]
    sc = prim.repeat(reps)
    sc.calc = calc
    sc.set_constraint(FixCom())
    sites = sc.get_positions().copy()          # relaxed lattice sites

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(sc, temperature_K=temperature, rng=rng)
    Stationary(sc)
    dyn = Langevin(sc, timestep * units.fs, temperature_K=temperature,
                   friction=0.01 / units.fs, fixcm=False, rng=rng)

    n_equil = int(steps * equil_frac)
    print(f"  MD: {len(sc)} atoms ({'×'.join(map(str, reps))}), {temperature} K, "
          f"{steps} steps of {timestep} fs ({n_equil} equilibration)")
    dyn.run(n_equil)

    # Production: accumulate positions in N_STAT blocks; the spread of the block
    # means estimates the statistical error of the average structure.
    N_STAT = 5
    block_sum = np.zeros((N_STAT,) + sites.shape)
    block_n = np.zeros(N_STAT)
    total_sq = np.zeros_like(sites)
    gammas = []
    n_samples = max(N_STAT, (steps - n_equil) // sample_every)
    for s in range(n_samples):
        dyn.run(sample_every)
        p = sc.get_positions()
        b = min(N_STAT - 1, s * N_STAT // n_samples)
        block_sum[b] += p
        block_n[b] += 1
        total_sq += p * p
        if calm_ok and s % max(1, n_samples // 10) == 0:
            gammas.append(calm_gamma(sc, calc)[0])

    block_mean = block_sum / block_n[:, None, None]
    mean = block_mean.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum(total_sq / block_n.sum() - mean ** 2, axis=1))))
    sem = np.linalg.norm(block_mean.std(axis=0, ddof=1), axis=1) / np.sqrt(N_STAT)
    noise = float(sem.max())
    drift = np.linalg.norm(mean - sites, axis=1)

    d = prim.get_all_distances(mic=True)
    d_nn = float(d[d > 1e-6].min()) if len(prim) > 1 else float(heights.min())
    diffusing = bool(drift.max() > 0.5 * d_nn)

    # the average structure is only resolved to ~noise, so detect symmetry above it
    symprec_used = max(symprec_avg, 3.0 * noise)
    too_noisy = symprec_used > 0.25

    avg = Atoms(numbers=sc.numbers, cell=sc.cell, positions=mean, pbc=True)
    avg.wrap()
    out = {
        "temperature_K": temperature,
        "supercell": reps,
        "rms_displacement_A": rms,
        "position_noise_A": noise,
        "symprec_used_A": symprec_used,
        "too_noisy": too_noisy,
        "max_mean_position_shift_A": float(drift.max()),
        "nearest_neighbour_A": d_nn,
        "diffusing": diffusing,
        "avg_sg_number": None,
        "avg_sg_symbol": None,
        "verdict_avg": None,
        "gamma_max_md": max(gammas) if gammas else None,
    }
    if not diffusing and not too_noisy:
        sg, sym, _ = space_group(avg, symprec_used)
        ver = filling_verdict(avg, symprec=symprec_used, mode=mode)
        out.update(avg_sg_number=sg, avg_sg_symbol=sym,
                   verdict_avg=ver["verdict"] if ver else None)
        write("POSCAR_md_average", avg, format="vasp", direct=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# DFT hand-off
# ──────────────────────────────────────────────────────────────────────────────

def write_dft_handoff(prim, label, info, outdir="dft_handoff"):
    """VASP + SOC inputs and an IrRep script for checking band touchings with DFT."""
    d = os.path.join(outdir, label)
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "POSCAR"), prim, format="vasp", direct=True)

    system = (f"{label} (SG #{info.get('sg_number')} {info.get('sg_symbol')}, "
              f"verdict {info.get('verdict')})")
    scf = f"""# VASP SCF with spin-orbit coupling (written by vasp-grace grace-symmetry)
# Check ENCUT against your POTCARs.
SYSTEM = {system}
PREC = Accurate
ENCUT = 520
EDIFF = 1E-7
ISMEAR = 0
SIGMA = 0.05
LSORBIT = .TRUE.
LNONCOLLINEAR = .TRUE.
ISYM = 2
MAGMOM = {3 * len(prim)}*0.0
LMAXMIX = 4
LWAVE = .TRUE.
LCHARG = .TRUE.
"""
    bands = scf.replace("LWAVE = .TRUE.\nLCHARG = .TRUE.\n",
                        "ICHARG = 11\nLWAVE = .TRUE.\nLCHARG = .FALSE.\n"
                        "# NBANDS: leave to VASP or set >= NELECT + 16 (from the SCF OUTCAR)\n")
    with open(os.path.join(d, "INCAR.scf"), "w") as f:
        f.write(scf)
    with open(os.path.join(d, "INCAR.bands"), "w") as f:
        f.write(bands)
    with open(os.path.join(d, "KPOINTS.scf"), "w") as f:
        f.write("Automatic Gamma-centred mesh\n0\nAuto\n  40\n")

    path = prim.cell.bandpath()
    lines = [f"High-symmetry path for {label}", "40", "Line-mode", "Reciprocal"]
    for segment in path.path.split(","):
        labels = [segment[i:i + 1] for i in range(len(segment))]
        # combine primed / numbered labels such as "X1"
        merged = []
        for c in labels:
            if merged and (c.isdigit() or c == "'"):
                merged[-1] += c
            else:
                merged.append(c)
        for a, b in zip(merged[:-1], merged[1:]):
            ka, kb = path.special_points[a], path.special_points[b]
            lines.append(f"  {ka[0]:.6f}  {ka[1]:.6f}  {ka[2]:.6f}   ! {a}")
            lines.append(f"  {kb[0]:.6f}  {kb[1]:.6f}  {kb[2]:.6f}   ! {b}")
            lines.append("")
    with open(os.path.join(d, "KPOINTS.bands"), "w") as f:
        f.write("\n".join(lines))

    irrep = f"""#!/usr/bin/env bash
# Little-group irreps of the SOC bands, to confirm symmetry-enforced touchings.
# Run after the band calculation (WAVECAR from INCAR.bands). Requires: pip install irrep
set -euo pipefail
NELECT=$(grep -m1 NELECT OUTCAR | awk '{{print int($3)}}')
# go past the Fermi level: a symmetry-enforced touching sits exactly at NELECT
IBEND=$((NELECT + 8))
irrep -code=vasp -fWAV=WAVECAR -fPOS=POSCAR -Ecut=100 -IBstart=1 -IBend=$IBEND -spinor > irrep_output.txt
echo "Irreps for {label} written to irrep_output.txt (bands 1-$IBEND, NELECT=$NELECT)"
"""
    p = os.path.join(d, "run_irrep.sh")
    with open(p, "w") as f:
        f.write(irrep)
    os.chmod(p, 0o755)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def assess(report):
    """Plain-language conclusions from the collected results."""
    lines = []
    v = report.get("verdict_relaxed")
    if v is None:
        lines.append("Band-insulator verdict unavailable: install symlint "
                     "(pip install -e \".[condmat]\").")
    elif v["verdict"] == "allowed":
        lines.append(f"ν = {v['nu']} is allowed in {v['sg_symbol']}: symmetry permits a band "
                     "insulator, so there is no filling-enforced protection to test.")
    else:
        lines.append(f"ν = {v['nu']} is forbidden in {v['sg_symbol']} ({v['verdict']}): "
                     "the relaxed crystal cannot be a band insulator.")
        gapping = [r for r in report["soft_modes"] if r["verdict_distorted"] == "allowed"]
        if not report["soft_modes"]:
            lines.append("No imaginary phonons at the commensurate q-points: the protected "
                         "structure is dynamically stable at this level.")
        elif gapping:
            qs = sorted({"(" + ", ".join(f"{x:.2f}" for x in r["q"]) + ")" for r in gapping})
            lines.append(f"Soft modes at q = {', '.join(qs)} lead to distorted "
                         "structures where a band insulator is allowed: the crystal can "
                         "remove the protection by distorting (Peierls/CDW-type).")
        else:
            lines.append("Every soft-mode distortion keeps the filling forbidden: the "
                         "protection survives the lattice instabilities found.")
        md = report.get("md")
        if md:
            if md["diffusing"]:
                lines.append(f"At {md['temperature_K']} K some atoms diffuse; the average "
                             "structure is not meaningful.")
            elif md["too_noisy"]:
                lines.append(f"At {md['temperature_K']} K the average positions are only "
                             f"resolved to {md['position_noise_A']:.2f} Å; run longer "
                             "(--md_steps) before reading a space group from them.")
            elif md["verdict_avg"] is not None:
                same = md["verdict_avg"] in PROTECTED
                lines.append(f"At {md['temperature_K']} K the average structure is "
                             f"{md['avg_sg_symbol']} (tolerance {md['symprec_used_A']:.2f} Å) "
                             f"and the filling is {'still forbidden' if same else 'allowed'}.")
    c = report["chirality_relaxed"]
    if c["chiral"]:
        lines.append(f"The structure is chiral ({c['sg_symbol']}); "
                     + (f"its enantiomer is {c['enantiomer_sg_symbol']}."
                        if c["enantiomorphic_pair"]
                        else "both enantiomers share the space group."))
    g = report.get("gamma_relaxed")
    if g is not None:
        lines.append(f"CALM max γ = {g:.2f} for the relaxed cell "
                     + ("(inside the training distribution)." if g <= 1
                        else "(extrapolating: confirm energies with DFT)."))
    return lines


def write_reports(report, soft_rows):
    with open("symmetry_report.json", "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    with open("soft_modes.dat", "w") as f:
        f.write(f"# {'q':>20}  {'freq(THz)':>10}  {'deg':>4}  {'trial':>8}  "
                f"{'SG':>4}  {'symbol':>12}  {'nu':>6}  {'verdict':>18}  {'max_γ':>7}\n")
        for r in soft_rows:
            q = "(" + ", ".join(f"{x:.3f}" for x in r["q"]) + ")"
            gm = f"{r['gamma_max']:.3f}" if r["gamma_max"] is not None else "-"
            f.write(f"  {q:>20}  {r['frequency_THz']:10.3f}  {r['degeneracy']:4d}  "
                    f"{r['trial']:>8}  {r['distorted_sg_number']!s:>4}  "
                    f"{r['distorted_sg_symbol']!s:>12}  {r['nu_distorted']!s:>6}  "
                    f"{r['verdict_distorted']!s:>18}  {gm:>7}\n")

    def ver(v):
        return "n/a (symlint not installed)" if v is None else \
            f"{v['verdict']}  (SG #{v['sg_number']} {v['sg_symbol']}, ν = {v['nu']}, rule {v['allowed_rule']})"

    with open("symmetry_report.txt", "w") as f:
        f.write("grace-symmetry report\n=====================\n\n")
        f.write(f"Structure      : {report['formula']} ({report['input']})\n")
        f.write(f"Model          : {report['model']}\n")
        f.write(f"Rule mode      : {report['rule_mode']}\n\n")
        f.write(f"Input          : {ver(report['verdict_input'])}\n")
        f.write(f"Relaxed        : {ver(report['verdict_relaxed'])}\n")
        if report["relax"]:
            r = report["relax"]
            f.write(f"Relaxation     : {'converged' if r['converged'] else 'NOT converged'} "
                    f"in {r['steps']} steps (fmax {r['fmax']} eV/Å, space group held)\n")
        c = report["chirality_relaxed"]
        f.write(f"Chirality      : {'chiral' if c['chiral'] else 'achiral'}"
                + (f", enantiomer {c['enantiomer_sg_symbol']}, handedness {c['handedness']}"
                   if c["chiral"] else "") + "\n")
        g = report.get("gamma_relaxed")
        f.write(f"CALM max γ     : {g:.3f}\n" if g is not None else
                "CALM max γ     : n/a (model has no built-in uncertainty)\n")
        f.write(f"\nPhonons        : {report['phonons']['n_qpoints']} q-points of a "
                f"{report['phonons']['supercell']}³ supercell\n")
        if report["phonons"]["lowest_stable"]:
            q, nu = report["phonons"]["lowest_stable"]
            qs = "(" + ", ".join(f"{x:.3f}" for x in q) + ")"
            f.write(f"Lowest real mode: {nu:.3f} THz at q = {qs}\n")
        f.write(f"Soft-mode trials: {len(soft_rows)} (see soft_modes.dat)\n")
        if report.get("md"):
            m = report["md"]
            f.write(f"\nMD             : {m['temperature_K']} K, supercell {m['supercell']}, "
                    f"rms displacement {m['rms_displacement_A']:.3f} Å, "
                    f"position noise {m['position_noise_A']:.3f} Å\n")
            if m["diffusing"]:
                f.write("Average SG     : not meaningful (diffusion detected)\n")
            elif m["too_noisy"]:
                f.write("Average SG     : not resolved (average too noisy; increase --md_steps)\n")
            else:
                f.write(f"Average SG     : {m['avg_sg_symbol']} (#{m['avg_sg_number']}) at "
                        f"tolerance {m['symprec_used_A']:.3f} Å, verdict {m['verdict_avg']}\n")
            if m["gamma_max_md"] is not None:
                f.write(f"CALM max γ (MD): {m['gamma_max_md']:.3f}\n")
        f.write("\nAssessment\n----------\n")
        for line in report["assessment"]:
            f.write(f"- {line}\n")
        f.write("\nReference: H. Watanabe, H. C. Po, M. P. Zaletel, A. Vishwanath, "
                "Phys. Rev. Lett. 117, 096404 (2016).\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether a crystal's band-insulator verdict (filling rule, via symlint) "
            "survives GRACE relaxation, soft phonons and temperature. Also reports chirality."
        )
    )
    parser.add_argument("--poscar", default="POSCAR", help="Input structure (default: POSCAR).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"GRACE model name or path (default: {DEFAULT_MODEL}).")
    parser.add_argument("--spinless", action="store_true",
                        help="Apply the spinless rule instead of the SOC (spinful) one.")
    parser.add_argument("--symprec", type=float, default=1e-3,
                        help="spglib tolerance for the 0 K structures (default: 1e-3).")
    parser.add_argument("--no_relax", action="store_true", help="Skip the GRACE relaxation.")
    parser.add_argument("--fmax", type=float, default=0.002,
                        help="Relaxation force criterion in eV/Å (default: 0.002).")
    parser.add_argument("--relax_steps", type=int, default=300, help="Max relaxation steps.")
    parser.add_argument("--supercell", type=int, default=2,
                        help="Phonon supercell n (n×n×n); sets the commensurate q-points (default: 2).")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Finite displacement in Å (default: 0.01).")
    parser.add_argument("--imag_tol", type=float, default=0.3,
                        help="Modes below -imag_tol THz count as soft (default: 0.3).")
    parser.add_argument("--amplitude", type=float, default=0.05,
                        help="Max displacement along a soft mode in Å (default: 0.05).")
    parser.add_argument("--md", type=float, metavar="T",
                        help="Also run NVT MD at T kelvin and analyse the average structure.")
    parser.add_argument("--md_steps", type=int, default=4000, help="MD steps (default: 4000).")
    parser.add_argument("--timestep", type=float, default=2.0, help="MD time step in fs (default: 2).")
    parser.add_argument("--md_min_length", type=float, default=10.0,
                        help="Minimum MD supercell height in Å (default: 10).")
    parser.add_argument("--md_symprec", type=float, default=0.05,
                        help="spglib tolerance for the MD average structure (default: 0.05).")
    parser.add_argument("--handoff", action="store_true",
                        help="Write VASP + SOC inputs and an IrRep script to dft_handoff/.")
    parser.add_argument("--keep_phonon_cache", action="store_true",
                        help="Keep the finite-displacement force files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for MD (default: 42).")
    args = parser.parse_args()

    if not os.path.exists(args.poscar):
        print(f"Error: {args.poscar} not found."); sys.exit(1)
    mode = "spinless" if args.spinless else "soc"

    atoms = read(args.poscar)
    print(f"Loaded {atoms.get_chemical_formula()} ({len(atoms)} atoms) from {args.poscar}")
    if importlib.util.find_spec("symlint") is None:
        print("Note: symlint not installed; verdicts will be reported as n/a.")

    calc = load_calculator(args.model)
    try:
        enable_calm(calc, args.model)
        calc.disable_uq()                  # only on for single evaluations
        calm_ok = True
    except ValueError:
        calm_ok = False

    report = {
        "input": args.poscar,
        "formula": atoms.get_chemical_formula(),
        "model": args.model,
        "rule_mode": mode,
        "verdict_input": filling_verdict(atoms, args.symprec, mode),
        "chirality_input": chirality_report(atoms, args.symprec),
        "relax": None,
    }

    # 1. Relax with the space group held, then standardize
    if args.no_relax:
        relaxed = atoms
    else:
        print("Relaxing with the space group held fixed...")
        relaxed, converged, steps = relax_symmetric(atoms, calc, args.fmax, args.relax_steps,
                                                    args.symprec)
        report["relax"] = {"converged": converged, "steps": steps, "fmax": args.fmax}
    prim = standard_primitive(relaxed, args.symprec)
    write("POSCAR_primitive", prim, format="vasp", direct=True)
    report["verdict_relaxed"] = filling_verdict(prim, args.symprec, mode)
    report["chirality_relaxed"] = chirality_report(prim, args.symprec)
    report["gamma_relaxed"] = calm_gamma(prim, calc)[0] if calm_ok else None
    sg, sym, _ = space_group(prim, args.symprec)
    print(f"Relaxed primitive cell: {len(prim)} atoms, {sym} (#{sg})")

    # 2. Phonons at Γ and the zone boundary; follow soft modes
    workdir = tempfile.mkdtemp(prefix="grace_symmetry_", dir=".")
    try:
        print(f"Phonons on a {args.supercell}×{args.supercell}×{args.supercell} supercell...")
        qs, freqs, modes = phonons_at(prim, calc, args.supercell, args.delta, workdir)
    finally:
        if not args.keep_phonon_cache:
            shutil.rmtree(workdir, ignore_errors=True)
    soft_rows, lowest = analyse_soft_modes(
        prim, qs, freqs, modes, args.supercell, args.imag_tol, args.amplitude,
        args.symprec, mode, calc, calm_ok)
    report["phonons"] = {
        "supercell": args.supercell,
        "n_qpoints": len(qs),
        "qpoints": qs,
        "frequencies_THz": freqs,
        "lowest_stable": lowest,
    }
    report["soft_modes"] = soft_rows
    n_soft = len({(tuple(r["q"]), round(r["frequency_THz"], 3)) for r in soft_rows})
    print(f"  {n_soft} soft mode(s) below -{args.imag_tol} THz, {len(soft_rows)} distortion trials")

    # 3. Optional finite-temperature average structure
    if args.md is not None:
        report["md"] = md_average_structure(
            prim, calc, args.md, args.md_steps, args.timestep, 0.3, args.md_min_length,
            sample_every=10, seed=args.seed, symprec_avg=args.md_symprec, mode=mode,
            calm_ok=calm_ok)

    # 4. Assessment, reports, DFT hand-off
    report["assessment"] = assess(report)
    write_reports(report, soft_rows)
    if args.handoff:
        info = report["verdict_relaxed"] or {"sg_number": sg, "sg_symbol": sym}
        label = f"{prim.get_chemical_formula()}_SG{sg}"
        print(f"DFT inputs written to {write_dft_handoff(prim, label, info)}/")

    print("\nAssessment")
    for line in report["assessment"]:
        print(f"  - {line}")
    print("\nWritten: symmetry_report.txt, symmetry_report.json, soft_modes.dat, POSCAR_primitive")


if __name__ == "__main__":
    main()
