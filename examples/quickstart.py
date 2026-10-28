import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # vasp-grace quickstart

    An interactive tour of **vasp-grace**: GRACE machine-learning potentials driven
    through ASE, plus a few of the analysis modules that ship with the package.

    Run it with

    ```bash
    pip install -e ".[examples]"
    marimo edit examples/quickstart.py
    ```

    The first GRACE call downloads the foundation model, so it can take a minute.
    Everything below re-runs automatically when you change a control.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from ase.build import bulk
    from ase.eos import EquationOfState
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS
    from ase.units import GPa

    from vasp_grace.calculator import DEFAULT_MODEL, available_models, load_calculator
    from vasp_grace.condmat import moire, topology
    return (
        DEFAULT_MODEL,
        available_models,
        EquationOfState,
        FrechetCellFilter,
        GPa,
        LBFGS,
        bulk,
        load_calculator,
        mo,
        moire,
        np,
        plt,
        topology,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Pick a crystal and a GRACE model

    Any model from the GRACE foundation-model family works: `FS`, `1L`, `2L` and
    `3L` architectures trained on OAM, OMAT or SMAX data. The list below comes from
    your installed `tensorpotential`; `grace_models list` shows the same names.
    A path to your own fitted model overrides the list. Either value is what you
    would put in `GRACE_MODEL` in an `INCAR` for `vasp-grace`.
    """)
    return


@app.cell
def _(DEFAULT_MODEL, available_models, mo):
    element = mo.ui.dropdown(
        options=["Cu", "Al", "Ni", "Ag", "Au", "Si", "Mo", "W"],
        value="Cu",
        label="Element (ASE reference structure)",
    )
    model_choice = mo.ui.dropdown(
        options=[m for m in available_models() if not m.endswith("-fp64")],
        value=DEFAULT_MODEL,
        label="GRACE foundation model",
        searchable=True,
    )
    model_path = mo.ui.text(
        placeholder="optional: path to your own GRACE model", full_width=True
    ).form(submit_button_label="Use this model")
    mo.vstack([element, model_choice, model_path])
    return element, model_choice, model_path


@app.cell
def _(load_calculator, model_choice, model_path):
    model_name = (model_path.value or "").strip() or model_choice.value
    calc = load_calculator(model_name)
    return calc, model_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Single-point evaluation

    What `vasp-grace` does for `NSW = 0`: energy, forces and stress for the input cell.
    """)
    return


@app.cell
def _(GPa, bulk, calc, element, mo, model_name, np):
    atoms = bulk(element.value)
    atoms.calc = calc
    _energy = atoms.get_potential_energy()
    _fmax = np.abs(atoms.get_forces()).max()
    _stress = atoms.get_stress() / GPa

    mo.md(f"""
    | Quantity | Value |
    |---|---|
    | Model | `{model_name}` |
    | Structure | {atoms.get_chemical_formula()}, {len(atoms)} atom(s), V = {atoms.get_volume():.3f} Å³ |
    | Energy | {_energy:.4f} eV ({_energy / len(atoms):.4f} eV/atom) |
    | max \\|F\\| | {_fmax:.2e} eV/Å |
    | Stress (xx, yy, zz) | {_stress[0]:.2f}, {_stress[1]:.2f}, {_stress[2]:.2f} GPa |
    """)
    return (atoms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Equation of state

    Scale the cell isotropically and fit a Birch–Murnaghan EOS to get the
    equilibrium volume and bulk modulus.
    """)
    return


@app.cell
def _(mo):
    strain = mo.ui.slider(
        0.02, 0.12, step=0.01, value=0.06, label="Max volumetric strain (±)", show_value=True
    )
    strain
    return (strain,)


@app.cell
def _(EquationOfState, GPa, atoms, calc, mo, np, plt, strain):
    _volumes, _energies = [], []
    for _scale in np.linspace(1 - strain.value, 1 + strain.value, 9):
        _a = atoms.copy()
        _a.set_cell(atoms.cell * _scale ** (1 / 3), scale_atoms=True)
        _a.calc = calc
        _volumes.append(_a.get_volume())
        _energies.append(_a.get_potential_energy())

    _eos = EquationOfState(_volumes, _energies, eos="birchmurnaghan")
    _v0, _e0, _B = _eos.fit()

    _fig, _ax = plt.subplots(figsize=(5, 3.5))
    _v_fine = np.linspace(min(_volumes), max(_volumes), 200)
    _ax.plot(_v_fine, _eos.func(_v_fine, *_eos.eos_parameters), lw=1.5)
    _ax.plot(_volumes, _energies, "o")
    _ax.axvline(_v0, ls="--", lw=0.8, color="gray")
    _ax.set_xlabel("Volume (Å³)")
    _ax.set_ylabel("Energy (eV)")
    _fig.tight_layout()

    mo.hstack(
        [
            _fig,
            mo.md(f"""
            **V₀** = {_v0:.3f} Å³
            **E₀** = {_e0:.4f} eV
            **B₀** = {_B / GPa:.1f} GPa
            """),
        ],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Full cell relaxation

    The equivalent of `IBRION = 1, ISIF = 3` in `vasp-grace`: LBFGS on a
    `FrechetCellFilter`, so positions and cell relax together.
    """)
    return


@app.cell
def _(FrechetCellFilter, LBFGS, atoms, calc, mo):
    relaxed = atoms.copy()
    relaxed.calc = calc
    _opt = LBFGS(FrechetCellFilter(relaxed), logfile=None)
    _converged = _opt.run(fmax=0.005, steps=200)

    _a0 = atoms.cell.cellpar()
    _a1 = relaxed.cell.cellpar()
    mo.md(f"""
    Converged: **{_converged}** after {_opt.get_number_of_steps()} steps

    | | a (Å) | b (Å) | c (Å) | V (Å³) |
    |---|---|---|---|---|
    | Input   | {_a0[0]:.4f} | {_a0[1]:.4f} | {_a0[2]:.4f} | {atoms.get_volume():.3f} |
    | Relaxed | {_a1[0]:.4f} | {_a1[1]:.4f} | {_a1[2]:.4f} | {relaxed.get_volume():.3f} |
    """)
    return (relaxed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Topological fingerprint (`grace-topology`)

    Persistent-homology Betti curves of the relaxed crystal versus a thermally
    rattled copy. Order shows up as sharp steps in β₀ and β₁; disorder smears them.
    No GRACE call is needed here.
    """)
    return


@app.cell
def _(mo):
    rattle = mo.ui.slider(
        0.0, 0.3, step=0.02, value=0.1, label="Rattle amplitude σ (Å)", show_value=True
    )
    rmax = mo.ui.slider(2.0, 6.0, step=0.5, value=4.0, label="r_max (Å)", show_value=True)
    mo.hstack([rattle, rmax])
    return rattle, rmax


@app.cell
def _(np, plt, rattle, relaxed, rmax, topology):
    _crystal = relaxed.repeat((3, 3, 3))
    _crystal.calc = None
    _disordered = _crystal.copy()
    _disordered.rattle(stdev=rattle.value, seed=0)
    _r = np.linspace(0.5, rmax.value, 80)

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for _label, _at in [("crystal", _crystal), (f"rattled σ={rattle.value:.2f} Å", _disordered)]:
        _D = topology.pbc_distance_matrix(_at.positions, np.array(_at.cell))
        _b0, _b1, _b2, _pairs, _inf = topology.compute_betti_curves_scipy(_D, _r)
        _axes[0].plot(_r, _b0, label=_label)
        _axes[1].plot(_r, _b1, label=_label)
    _axes[0].set_ylabel("β₀ (components)")
    _axes[1].set_ylabel("β₁ (cycles)")
    for _ax in _axes:
        _ax.set_xlabel("r (Å)")
    _axes[1].legend(frameon=False)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Commensurate moiré angles (`grace-moire --scan`)

    Twist angles for which a hexagonal bilayer has a finite periodic supercell,
    with the number of atoms per layer in that cell.
    """)
    return


@app.cell
def _(mo):
    m_max = mo.ui.slider(3, 15, step=1, value=8, label="m_max", show_value=True)
    m_max
    return (m_max,)


@app.cell
def _(m_max, mo, moire):
    _rows = [
        {"m": m, "n": n, "θ (deg)": round(theta, 3), "cells / layer": N, "L / a": round(L, 3)}
        for m, n, theta, N, L in moire.scan_csl_angles(m_max.value, lattice="hex")
    ]
    mo.ui.table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps on the command line

    | Task | Command |
    |---|---|
    | VASP drop-in (opt / MD / phonons / elastic) | `vasp-grace --poscar POSCAR --incar INCAR` |
    | Committee uncertainty | `grace-active-learning --poscar POSCAR --models m1 m2 m3` |
    | Metropolis MC with swaps | `grace-mc --poscar POSCAR --model GRACE-2L-OAM --swap Cl S` |
    | Dislocations | `grace-dislocation --poscar POSCAR --type screw --burgers 2.55` |
    | Thermal conductivity | `grace-kappa --poscar POSCAR --temperature 300` |
    | Phonon topology | `grace-topo-phonons --poscar POSCAR --berry` |
    """)
    return


if __name__ == "__main__":
    app.run()
