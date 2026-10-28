"""
chirality.py

Structural chirality from space-group symmetry.

A crystal is chiral when its space group contains only proper operations
(rotations and screw axes, det R = +1): the 65 Sohncke groups. Inverting a
chiral structure gives its enantiomer, which cannot be superimposed on it.

For 22 Sohncke groups (11 enantiomorphic pairs such as P3_121 / P3_221 for
trigonal Te) the two enantiomers have different space-group symbols, so the
symbol fixes the handedness. For the other 43 (e.g. P2_13 for CoSi, the host of
Kramers-Weyl and multifold fermions) both hands share one space group and the
handedness has to be read from the atomic arrangement.
"""

import numpy as np
import spglib


def _cell(atoms):
    return (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)


def _dataset(cell, symprec):
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:
        raise ValueError(f"spglib could not determine the space group (symprec={symprec}).")
    return ds


def chirality_report(atoms, symprec=1e-3):
    """
    Classify the structural chirality of `atoms`.

    Returns a dict with:
        sg_number, sg_symbol      space group of the structure
        chiral                    True for a Sohncke group (proper operations only)
        enantiomer_sg_number/_symbol
                                  space group of the inverted structure (chiral only)
        enantiomorphic_pair       True if the two hands have different space groups
        handedness                the space-group symbol when it fixes the hand,
                                  otherwise a note that it does not
    """
    lattice, scaled, numbers = _cell(atoms)
    ds = _dataset((lattice, scaled, numbers), symprec)
    chiral = all(round(np.linalg.det(r)) == 1 for r in ds.rotations)

    out = {
        "sg_number": int(ds.number),
        "sg_symbol": str(ds.international),
        "chiral": bool(chiral),
        "enantiomer_sg_number": None,
        "enantiomer_sg_symbol": None,
        "enantiomorphic_pair": False,
        "handedness": None,
    }
    if not chiral:
        return out

    inverted = _dataset((lattice, (-scaled) % 1.0, numbers), symprec)
    out["enantiomer_sg_number"] = int(inverted.number)
    out["enantiomer_sg_symbol"] = str(inverted.international)
    out["enantiomorphic_pair"] = int(inverted.number) != int(ds.number)
    out["handedness"] = (
        str(ds.international) if out["enantiomorphic_pair"]
        else "not fixed by the space group (both hands share it)"
    )
    return out
