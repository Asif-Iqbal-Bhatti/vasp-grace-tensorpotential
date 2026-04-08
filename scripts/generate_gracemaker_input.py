#!/usr/bin/env python3
"""
generate_gracemaker_input.py
============================
Generate a ``input.yaml`` configuration file for gracemaker GRACE MLIP
training, covering GRACE/FS, GRACE-1L, and GRACE-2L architectures.

The generated YAML can be passed directly to ``gracemaker input.yaml``
or fine-tuned manually before training.

Usage
-----
  python generate_gracemaker_input.py \\
      --train-file  collected.pkl.gz \\
      --model-type  GRACE-2L \\
      --complexity  medium \\
      --elements    Li Ni Mn O \\
      --cutoff      6.0 \\
      --output      input.yaml

  # Fine-tune a foundation model
  python generate_gracemaker_input.py \\
      --train-file  collected.pkl.gz \\
      --finetune    GRACE-2L-OMAT-medium \\
      --elements    Li Ni Mn O \\
      --output      finetune_input.yaml

Author: vasp-grace workflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gen_input")

# ── Model presets ─────────────────────────────────────────────────────────────
# Tuned defaults for each GRACE model family / complexity combination.

MODEL_PRESETS = {
    "GRACE/FS": {
        "small":  {"cutoff": 5.0, "batch_size": 32, "lr": 5e-4, "updates": 5_000},
        "medium": {"cutoff": 5.5, "batch_size": 16, "lr": 5e-4, "updates": 10_000},
        "large":  {"cutoff": 6.0, "batch_size":  8, "lr": 2e-4, "updates": 20_000},
    },
    "GRACE-1L": {
        "small":  {"cutoff": 5.0, "batch_size": 16, "lr": 5e-4, "updates": 10_000},
        "medium": {"cutoff": 6.0, "batch_size":  8, "lr": 2e-4, "updates": 20_000},
        "large":  {"cutoff": 6.0, "batch_size":  4, "lr": 1e-4, "updates": 30_000},
    },
    "GRACE-2L": {
        "small":  {"cutoff": 5.0, "batch_size":  8, "lr": 5e-4, "updates": 15_000},
        "medium": {"cutoff": 6.0, "batch_size":  4, "lr": 2e-4, "updates": 25_000},
        "large":  {"cutoff": 6.0, "batch_size":  2, "lr": 1e-4, "updates": 50_000},
    },
}

FOUNDATION_MODELS = [
    "GRACE-2L-OMAT-medium",
    "GRACE-2L-OMAT-small",
    "GRACE-2L-SMAX-medium",
    "GRACE-2L-MPtrj-medium",
    "GRACE-1L-OMAT-medium",
]

# Loss weight recommendations per model family
LOSS_WEIGHTS = {
    "GRACE/FS":  {"energy": 1.0, "forces": 1.0,  "stress": 0.01},
    "GRACE-1L": {"energy": 1.0, "forces": 1.0,  "stress": 0.01},
    "GRACE-2L": {"energy": 16.,  "forces": 32.,  "stress": 128.},
}


# ── YAML builders ─────────────────────────────────────────────────────────────

def _base_config(
    train_file: str,
    test_file: Optional[str],
    test_fraction: float,
    elements: List[str],
    cutoff: float,
    batch_size: int,
    learning_rate: float,
    n_updates: int,
    energy_weight: float,
    forces_weight: float,
    stress_weight: float,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    seed: int,
) -> dict:
    """Return the common configuration block shared by all model types."""
    dataset_block: dict = {
        "filename": train_file,
        "energy_key": energy_key,
        "forces_key": forces_key,
        "virial_key": stress_key,
    }
    if test_file:
        dataset_block["test_filename"] = test_file
    else:
        dataset_block["test_size"] = test_fraction

    return {
        "seed": seed,
        "cutoff": cutoff,
        "elements": elements,
        "dataset": dataset_block,
        "fit": {
            "optimizer": "Adam",
            "loss": {
                "type": "huber",
                "huber_delta": 0.01,
                "energy_weight": energy_weight,
                "forces_weight": forces_weight,
                "stress_weight": stress_weight,
            },
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "target_epochs": None,
            "target_total_updates": n_updates,
        },
    }


def build_grace_fs_yaml(
    train_file: str,
    test_file: Optional[str],
    test_fraction: float,
    elements: List[str],
    cutoff: float,
    complexity: str,
    batch_size: int,
    learning_rate: float,
    n_updates: int,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    seed: int,
) -> dict:
    """Build GRACE/FS (fast, local, C++ compatible) configuration."""
    w = LOSS_WEIGHTS["GRACE/FS"]
    cfg = _base_config(
        train_file, test_file, test_fraction, elements,
        cutoff, batch_size, learning_rate, n_updates,
        w["energy"], w["forces"], w["stress"],
        energy_key, forces_key, stress_key, seed,
    )
    cfg["model"] = {
        "preset": "GRACE_FS_latest",
        "complexity": complexity,
        # FS models can also export to YAML for native C++ LAMMPS
        "export_yaml": True,
    }
    return cfg


def build_grace_1l_yaml(
    train_file: str,
    test_file: Optional[str],
    test_fraction: float,
    elements: List[str],
    cutoff: float,
    complexity: str,
    batch_size: int,
    learning_rate: float,
    n_updates: int,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    seed: int,
) -> dict:
    """Build GRACE-1L (local, GPU-accelerated) configuration."""
    w = LOSS_WEIGHTS["GRACE-1L"]
    cfg = _base_config(
        train_file, test_file, test_fraction, elements,
        cutoff, batch_size, learning_rate, n_updates,
        w["energy"], w["forces"], w["stress"],
        energy_key, forces_key, stress_key, seed,
    )
    cfg["model"] = {
        "preset": "GRACE_1LAYER_latest",
        "complexity": complexity,
    }
    return cfg


def build_grace_2l_yaml(
    train_file: str,
    test_file: Optional[str],
    test_fraction: float,
    elements: List[str],
    cutoff: float,
    complexity: str,
    batch_size: int,
    learning_rate: float,
    n_updates: int,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    seed: int,
) -> dict:
    """Build GRACE-2L (semi-local, highest accuracy) configuration."""
    w = LOSS_WEIGHTS["GRACE-2L"]
    cfg = _base_config(
        train_file, test_file, test_fraction, elements,
        cutoff, batch_size, learning_rate, n_updates,
        w["energy"], w["forces"], w["stress"],
        energy_key, forces_key, stress_key, seed,
    )
    cfg["model"] = {
        "preset": "GRACE_2LAYER_latest",
        "complexity": complexity,
    }
    return cfg


def build_finetune_yaml(
    foundation_model: str,
    train_file: str,
    test_file: Optional[str],
    test_fraction: float,
    elements: List[str],
    batch_size: int,
    n_updates: int,
    energy_key: str,
    forces_key: str,
    stress_key: str,
    seed: int,
) -> dict:
    """Build a fine-tuning configuration starting from a foundation model."""
    w = LOSS_WEIGHTS["GRACE-2L"]
    # Fine-tune typically uses lower LR
    cfg = _base_config(
        train_file, test_file, test_fraction, elements,
        cutoff=6.0,  # inherited from foundation model; can be overridden
        batch_size=batch_size,
        learning_rate=1e-4,
        n_updates=n_updates,
        energy_weight=w["energy"],
        forces_weight=w["forces"],
        stress_weight=w["stress"],
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        seed=seed,
    )
    cfg["fit_type"] = "finetune"
    cfg["foundation_model"] = foundation_model
    # Switch loss weights mid-training (recommended for fine-tuning)
    cfg["fit"]["switch_weights"] = {
        "at_update": n_updates // 2,
        "energy_weight": w["forces"],   # increase energy weight after switch
        "forces_weight": w["forces"],
        "stress_weight": w["stress"],
    }
    return cfg


# ── Comment header ────────────────────────────────────────────────────────────

YAML_HEADER = dedent("""\
# =============================================================================
#  gracemaker input.yaml  –  generated by vasp-grace
#  Documentation: https://gracemaker.readthedocs.io
#
#  Run training with:
#      gracemaker input.yaml
#
#  Monitor with TensorBoard:
#      tensorboard --logdir seed/1/
#
#  Export final model:
#      gracemaker input.yaml -r -s
#
#  Use the trained model in ASE:
#      from tensorpotential.calculator.grace import GRACECalculator
#      calc = GRACECalculator("seed/1/saved_model")
#
#  Use in LAMMPS:
#      pair_style grace
#      pair_coeff * * seed/1/saved_model/ El1 El2 ...
# =============================================================================

""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data_group = parser.add_argument_group("Dataset")
    data_group.add_argument(
        "--train-file", required=True,
        help="Training dataset (pkl.gz or extxyz).",
    )
    data_group.add_argument(
        "--test-file", default=None,
        help="Separate test dataset. If omitted, a fraction of train is used.",
    )
    data_group.add_argument(
        "--test-fraction", type=float, default=0.05,
        help="Fraction of training data to use as test set. (default: 0.05)",
    )
    data_group.add_argument(
        "--energy-key", default="REF_energy",
    )
    data_group.add_argument(
        "--forces-key", default="REF_forces",
    )
    data_group.add_argument(
        "--stress-key", default="REF_virial",
    )

    model_group = parser.add_argument_group("Model")
    model_ex = model_group.add_mutually_exclusive_group(required=True)
    model_ex.add_argument(
        "--model-type", choices=["GRACE/FS", "GRACE-1L", "GRACE-2L"],
        help="GRACE model family to train from scratch.",
    )
    model_ex.add_argument(
        "--finetune", choices=FOUNDATION_MODELS, metavar="FOUNDATION_MODEL",
        help="Fine-tune a pre-trained foundation model. "
             f"Available: {', '.join(FOUNDATION_MODELS)}",
    )
    model_group.add_argument(
        "--complexity", choices=["small", "medium", "large"], default="medium",
        help="Model complexity tier. (default: medium)",
    )
    model_group.add_argument(
        "--cutoff", type=float, default=None,
        help="Cutoff radius in Å. Defaults to preset value for chosen model.",
    )
    model_group.add_argument(
        "--elements", nargs="+", required=True, metavar="El",
    )

    train_group = parser.add_argument_group("Training hyperparameters")
    train_group.add_argument("--batch-size",  type=int,   default=None)
    train_group.add_argument("--learning-rate", type=float, default=None)
    train_group.add_argument("--n-updates",   type=int,   default=None)
    train_group.add_argument("--seed",        type=int,   default=1)

    parser.add_argument(
        "--output", type=Path, default=Path("input.yaml"),
        help="Output YAML file. (default: input.yaml)",
    )
    args = parser.parse_args(argv)

    # ── Resolve preset / user overrides ──────────────────────────────────────
    if args.finetune:
        model_key = "GRACE-2L"
        preset = MODEL_PRESETS[model_key][args.complexity]
    else:
        model_key = args.model_type
        preset = MODEL_PRESETS[model_key][args.complexity]

    cutoff       = args.cutoff        or preset["cutoff"]
    batch_size   = args.batch_size    or preset["batch_size"]
    lr           = args.learning_rate or preset["lr"]
    n_updates    = args.n_updates     or preset["updates"]

    # ── Build config dict ─────────────────────────────────────────────────────
    common_kw = dict(
        train_file=args.train_file,
        test_file=args.test_file,
        test_fraction=args.test_fraction,
        elements=args.elements,
        batch_size=batch_size,
        n_updates=n_updates,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
        stress_key=args.stress_key,
        seed=args.seed,
    )

    if args.finetune:
        cfg = build_finetune_yaml(foundation_model=args.finetune, **common_kw)
    elif args.model_type == "GRACE/FS":
        cfg = build_grace_fs_yaml(cutoff=cutoff, complexity=args.complexity,
                                   learning_rate=lr, **common_kw)
    elif args.model_type == "GRACE-1L":
        cfg = build_grace_1l_yaml(cutoff=cutoff, complexity=args.complexity,
                                   learning_rate=lr, **common_kw)
    else:  # GRACE-2L
        cfg = build_grace_2l_yaml(cutoff=cutoff, complexity=args.complexity,
                                   learning_rate=lr, **common_kw)

    # ── Write YAML ────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    yaml_str = yaml.dump(cfg, default_flow_style=False, sort_keys=False)

    with open(args.output, "w") as fh:
        fh.write(YAML_HEADER)
        fh.write(yaml_str)

    log.info("Written: %s", args.output.resolve())
    log.info("")
    log.info("  Next steps:")
    log.info("    gracemaker %s", args.output)
    log.info("    tensorboard --logdir seed/1/   # monitor training")


if __name__ == "__main__":
    main()
