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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

DEFAULT_MODEL = "GRACE-2L-OAM"


def load_calculator(model_identifier=DEFAULT_MODEL, verbose=True):
    """Return an ASE calculator for a GRACE model path or foundation model name."""
    if os.path.exists(model_identifier):
        from tensorpotential.calculator import TPCalculator
        if verbose:
            print(f"Loading custom GRACE model from path: {model_identifier}")
        return TPCalculator(model_identifier)

    from tensorpotential.calculator.foundation_models import grace_fm
    if verbose:
        print(f"Loading GRACE foundation model: {model_identifier}")
    return grace_fm(model_identifier)
