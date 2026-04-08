#!/usr/bin/env python3
"""
tests/test_vasp_grace.py
========================
Unit tests for the vasp-grace workflow scripts.

Run with:
    pytest tests/ -v

No VASP or GRACE installation needed — all tests are lightweight and
use ASE's built-in test utilities or synthetic data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read, write


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def nacl_atoms():
    """Simple NaCl bulk for testing (no VASP needed)."""
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    return atoms


@pytest.fixture
def cu_atoms():
    """Cu FCC bulk — EMT supports Cu natively."""
    return bulk("Cu", crystalstructure="fcc", a=3.60)


@pytest.fixture
def emt_frames(cu_atoms):
    """List of frames with energies / forces from EMT on Cu bulk."""
    calc = EMT()
    frames = []
    for scale in np.linspace(0.95, 1.05, 5):
        a = cu_atoms.copy()
        a.set_cell(a.cell * scale, scale_atoms=True)
        a.calc = calc
        e = a.get_potential_energy()
        f = a.get_forces()
        a.info["REF_energy"] = e
        a.arrays["REF_forces"] = f
        frames.append(a)
    return frames


# ── vasp_to_extxyz tests ──────────────────────────────────────────────────────

class TestVaspToExtxyz:
    def test_selection_all(self, emt_frames, tmp_path):
        """_apply_selection('all') returns all frames."""
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
        from vasp_to_extxyz import _apply_selection
        selected = _apply_selection(emt_frames, "all")
        assert len(selected) == len(emt_frames)

    def test_selection_first(self, emt_frames):
        from vasp_to_extxyz import _apply_selection
        assert _apply_selection(emt_frames, "first") == [emt_frames[0]]

    def test_selection_last(self, emt_frames):
        from vasp_to_extxyz import _apply_selection
        assert _apply_selection(emt_frames, "last") == [emt_frames[-1]]

    def test_selection_first_and_last(self, emt_frames):
        from vasp_to_extxyz import _apply_selection
        result = _apply_selection(emt_frames, "first_and_last")
        assert result[0] is emt_frames[0]
        assert result[-1] is emt_frames[-1]

    def test_tag_frame_adds_keys(self, cu_atoms):
        from vasp_to_extxyz import _tag_frame
        atoms = cu_atoms.copy()
        atoms.calc = EMT()
        result = _tag_frame(
            atoms, "test", "REF_energy", "REF_forces", "REF_virial",
            "bulk", convert_stress=False,
        )
        assert result is not None
        assert "REF_energy" in result.info
        assert "REF_forces" in result.arrays
        assert result.info["config_type"] == "bulk"

    def test_convert_writes_extxyz(self, tmp_path, emt_frames):
        """End-to-end: write frames to extxyz and read back."""
        out = tmp_path / "test.extxyz"
        write(str(out), emt_frames, format="extxyz")
        loaded = read(str(out), index=":")
        assert len(loaded) == len(emt_frames)

    def test_energy_per_atom_reasonable(self, emt_frames):
        """Sanity: energies per atom are in a physical range for NaCl."""
        e_per_atom = [
            a.info["REF_energy"] / len(a) for a in emt_frames
        ]
        assert all(-20 < e < 20 for e in e_per_atom), f"Suspicious energies: {e_per_atom}"


# ── generate_gracemaker_input tests ──────────────────────────────────────────

class TestGenerateInput:
    def test_grace_fs_config_keys(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
        from generate_gracemaker_input import build_grace_fs_yaml
        cfg = build_grace_fs_yaml(
            train_file="train.pkl.gz",
            test_file=None,
            test_fraction=0.05,
            elements=["Li", "O"],
            cutoff=5.5,
            complexity="medium",
            batch_size=16,
            learning_rate=5e-4,
            n_updates=10_000,
            energy_key="REF_energy",
            forces_key="REF_forces",
            stress_key="REF_virial",
            seed=1,
        )
        assert "seed" in cfg
        assert "model" in cfg
        assert cfg["model"]["preset"] == "GRACE_FS_latest"
        assert cfg["elements"] == ["Li", "O"]

    def test_grace_2l_config(self):
        from generate_gracemaker_input import build_grace_2l_yaml
        cfg = build_grace_2l_yaml(
            train_file="train.pkl.gz",
            test_file=None,
            test_fraction=0.05,
            elements=["Li", "Ni", "Mn", "O"],
            cutoff=6.0,
            complexity="medium",
            batch_size=4,
            learning_rate=2e-4,
            n_updates=25_000,
            energy_key="REF_energy",
            forces_key="REF_forces",
            stress_key="REF_virial",
            seed=1,
        )
        assert cfg["model"]["preset"] == "GRACE_2LAYER_latest"
        assert cfg["fit"]["loss"]["energy_weight"] == 16.0

    def test_finetune_config(self):
        from generate_gracemaker_input import build_finetune_yaml
        cfg = build_finetune_yaml(
            foundation_model="GRACE-2L-OMAT-medium",
            train_file="train.pkl.gz",
            test_file=None,
            test_fraction=0.05,
            elements=["Li", "O"],
            batch_size=16,
            n_updates=10_000,
            energy_key="REF_energy",
            forces_key="REF_forces",
            stress_key="REF_virial",
            seed=1,
        )
        assert cfg["fit_type"] == "finetune"
        assert cfg["foundation_model"] == "GRACE-2L-OMAT-medium"

    def test_output_yaml_written(self, tmp_path):
        """CLI writes a valid YAML file."""
        import yaml
        from generate_gracemaker_input import main
        out = tmp_path / "input.yaml"
        main([
            "--train-file", "train.pkl.gz",
            "--model-type", "GRACE-2L",
            "--complexity", "small",
            "--elements", "Li", "O",
            "--output", str(out),
        ])
        assert out.exists()
        with open(out) as f:
            content = f.read()
        # Strip comment header before parsing
        yaml_lines = [l for l in content.splitlines() if not l.startswith("#")]
        cfg = yaml.safe_load("\n".join(yaml_lines))
        assert cfg["elements"] == ["Li", "O"]


# ── grace_lammps_input tests ──────────────────────────────────────────────────

class TestGraceLammpsInput:
    def test_pair_style_grace(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
        from grace_lammps_input import _pair_style_grace
        block = _pair_style_grace("/path/to/model", ["Li", "Ni", "Mn", "O"])
        assert "pair_style" in block
        assert "grace" in block
        assert "Li Ni Mn O" in block

    def test_pair_style_grace_fs(self):
        from grace_lammps_input import _pair_style_grace_fs
        block = _pair_style_grace_fs("/model.yaml", ["Li", "O"], active_set=None)
        assert "grace/fs" in block

    def test_pair_style_grace_fs_extrapolation(self):
        from grace_lammps_input import _pair_style_grace_fs
        block = _pair_style_grace_fs("/model.yaml", ["Li", "O"],
                                     active_set="/model.asi")
        assert "extrapolation" in block
        assert "grace_gamma" in block

    def test_lammps_data_written(self, tmp_path, nacl_atoms):
        """write_lammps_structure creates a readable data file."""
        from grace_lammps_input import write_lammps_structure
        out = tmp_path / "lammps.data"
        write_lammps_structure(nacl_atoms, out, elements=["Na", "Cl"])
        assert out.exists()
        assert out.stat().st_size > 0

    def test_cli_writes_files(self, tmp_path, nacl_atoms):
        """Full CLI run produces in.lammps and lammps.data."""
        # Write a temporary POSCAR
        poscar = tmp_path / "POSCAR"
        write(str(poscar), nacl_atoms, format="vasp")

        from grace_lammps_input import main
        out_dir = tmp_path / "lammps_run"
        main([
            "--structure", str(poscar),
            "--model",     "/fake/model",
            "--elements",  "Na", "Cl",
            "--ensemble",  "nvt",
            "--temperature", "300",
            "--steps",     "1000",
            "--output-dir", str(out_dir),
        ])
        assert (out_dir / "in.lammps").exists()
        assert (out_dir / "lammps.data").exists()

        script = (out_dir / "in.lammps").read_text()
        assert "grace" in script
        assert "Na Cl" in script


# ── validate_grace_model tests ────────────────────────────────────────────────

class TestValidate:
    def test_print_stats(self, capsys):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
        from validate_grace_model import print_stats
        ref  = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.1, 1.9, 3.1, 3.9])
        s = print_stats("Energy", ref, pred, "eV/atom")
        assert "rmse" in s
        assert s["rmse"] < 0.2

    def test_parity_plot_empty(self, tmp_path):
        from validate_grace_model import parity_plot
        # Should warn but not crash on empty arrays
        parity_plot(
            np.array([]), np.array([]),
            "x", "y", "title",
            tmp_path / "plot.png",
            "eV/atom",
        )

    def test_parity_plot_creates_file(self, tmp_path):
        from validate_grace_model import parity_plot
        ref  = np.random.randn(50)
        pred = ref + np.random.randn(50) * 0.1
        out  = tmp_path / "parity.png"
        parity_plot(ref, pred, "DFT", "GRACE", "Test", out, "eV/atom")
        assert out.exists()


# ── collect_vasp_dataset tests ────────────────────────────────────────────────

class TestCollect:
    def test_check_outcar_converged_false(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
        from collect_vasp_dataset import check_outcar_converged
        fake = tmp_path / "OUTCAR"
        fake.write_text("some random content\n")
        assert check_outcar_converged(fake) is False

    def test_check_outcar_converged_true(self, tmp_path):
        from collect_vasp_dataset import check_outcar_converged
        fake = tmp_path / "OUTCAR"
        fake.write_text("...reached required accuracy - stopping run\n")
        assert check_outcar_converged(fake) is True

    def test_build_free_atom_energy_str(self):
        from collect_vasp_dataset import build_free_atom_energy_str
        s = build_free_atom_energy_str(["Li", "Ni", "O"], {"Li": -1.9})
        assert "Li:-1.9" in s
        assert "Ni:auto" in s
        assert "O:auto"  in s

    def test_detect_elements(self, tmp_path):
        from collect_vasp_dataset import detect_elements
        # Write a fake POSCAR for NaCl
        nacl = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
        calc_dir = tmp_path / "calc_01"
        calc_dir.mkdir()
        write(str(calc_dir / "POSCAR"), nacl, format="vasp")
        elements = detect_elements([calc_dir])
        assert "Na" in elements
        assert "Cl" in elements
