"""Utility functions for atomic energy."""

import re
from typing import Literal, Optional

import numpy as np


def get_elements(system: str):
    """Return elements from system."""
    system = re.sub(r"[0-9]", "", system)
    begin = [iter1.span()[0] for iter1 in re.finditer(r"[A-Z]", system)]
    end = begin[1:] + [len(system)]
    elements = [system[b:e] for b, e in zip(begin, end)]
    return elements


def get_atomic_energies(
    elements: Optional[tuple] = None,
    formula: Optional[str] = None,
    functional: Literal["PBE", "PBEsol"] = "PBE",
    code: Literal["vasp"] = "vasp",
    return_dict: bool = False,
):
    """Return atomic energies."""
    pwd = __file__.replace("atomic_energies.py", "")
    if functional == "PBE" and code == "vasp":
        data = dict(np.loadtxt(pwd + "/energies_vasp_PBE.dat", dtype=str))
    elif functional == "PBEsol" and code == "vasp":
        data = dict(np.loadtxt(pwd + "/energies_vasp_PBEsol.dat", dtype=str))
    else:
        raise KeyError("No data for atomic energies:", code, functional)

    if formula is not None:
        elements = get_elements(formula)

    atom_e = [float(data[ele]) for ele in elements]
    if not return_dict:
        return atom_e, elements

    return dict(zip(elements, atom_e))


def get_atomic_energies_polymlp_in(
    elements: Optional[tuple] = None,
    formula: Optional[str] = None,
    functional: Literal["PBE", "PBEsol"] = "PBE",
    code: Literal["vasp"] = "vasp",
):
    """Return atomic energies in input file format."""
    atom_e, elements = get_atomic_energies(
        elements=elements, formula=formula, functional=functional, code=code
    )
    print("n_type", len(elements))
    print("elements", end="")
    for ele in elements:
        print("", ele, end="")
    print("")
    print("atomic_energy", end="")
    for e in atom_e:
        print("", e, end="")
    print("")
