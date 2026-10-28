"""
calculator.py

Shared GRACE calculator loading used by every vasp-grace module.

A model identifier is either a path to a fitted GRACE model on disk
(loaded with TPCalculator) or the name of a GRACE foundation model
such as GRACE-2L-OAM (loaded with grace_fm).
"""

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Take GPU memory as needed instead of reserving almost all of it at start-up,
# so several runs (or other GPU programs) can share the card.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

DEFAULT_MODEL = "GRACE-2L-OAM"


def available_models():
    """Names of the GRACE foundation models known to the installed tensorpotential."""
    try:
        from tensorpotential.calculator.foundation_models import MODELS_NAME_LIST
    except ImportError:  # older tensorpotential releases
        return [DEFAULT_MODEL]
    return list(MODELS_NAME_LIST)


def load_calculator(model_identifier=DEFAULT_MODEL, verbose=True, uq=False):
    """
    Return an ASE calculator for a GRACE model path or foundation model name.

    uq=True switches on GRACE's built-in CALM uncertainty: after each calculation
    calc.results["gamma"] holds the per-atom extrapolation grade (see enable_calm).
    """
    if os.path.exists(model_identifier):
        from tensorpotential.calculator import TPCalculator
        if verbose:
            print(f"Loading custom GRACE model from path: {model_identifier}")
        calc = TPCalculator(model_identifier)
    else:
        from tensorpotential.calculator.foundation_models import grace_fm
        if verbose:
            print(f"Loading GRACE foundation model: {model_identifier}")
        calc = grace_fm(model_identifier)

    if uq:
        enable_calm(calc, model_identifier)
    return calc


def enable_calm(calc, model_identifier=""):
    """
    Switch a GRACE calculator to CALM uncertainty mode.

    CALM gives each atom an extrapolation grade gamma = sigma / threshold, from the
    Mahalanobis distance of its local descriptor to the training-set clusters:
    gamma <= 1 means the environment is covered by the training data, gamma > 1
    means the model is extrapolating. One model is enough; no committee is needed.
    """
    modes = getattr(calc, "available_uq_modes", None) or []
    if not modes:
        raise ValueError(
            f"Model '{model_identifier}' does not provide CALM uncertainty. Use a model "
            "with UQ support (the UQ column of `grace_models list`, e.g. "
            "GRACE-2L-OMAT-medium-ft-AM) and tensorpotential >= 0.6."
        )
    calc.enable_uq("gamma_only" if "gamma_only" in modes else modes[0])
    return calc
