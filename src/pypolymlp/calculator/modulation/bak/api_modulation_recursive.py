"""API functions for modulation."""

import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.vasp_utils import write_poscar_file

from .modulation import PolymlpModulation
from .modulation_utils import (
    ModulationNode,
    eliminate_duplicate_structures,
    find_supercell_size_nearly_isotropic,
    optimize_geometry,
)


def run_modulations(
    properties: Properties,
    supercell_matrix: np.ndarray,
    unitcell: Optional[PolymlpStructure] = None,
    node: Optional[ModulationNode] = None,
    gtol: float = 1e-4,
    maxiter: int = 10000,
    pressure: float = 0.0,
    max_denom: int = 3,
    max_n_atom: int = 64,
    fc2: Optional[np.ndarray] = None,
    depth: int = 0,
    verbose: bool = False,
):
    """Run recursively calculations for modulated structures."""
    if node is not None:
        unitcell = node.structure

    mod = PolymlpModulation(
        properties=properties,
        supercell_matrix=supercell_matrix,
        unitcell=unitcell,
        pressure=pressure,
        gtol=gtol,
        maxiter=maxiter,
        verbose=verbose,
    )
    mod.run_phonon_and_modulations(max_denom=max_denom, fc2=fc2)
    node_parent = ModulationNode(
        mod.unitcell,
        mod.enthalpy,
        mod.spacegroup,
        is_imaginary=mod.is_imaginary,
        depth=depth,
    )
    print(mod.spacegroup)

    if not mod.is_imaginary:
        return node_parent

    # if verbose:
    print("Number of modulated structures:", len(mod.structure_children))
    print("- number of atoms (original):", len(unitcell.elements))

    depth += 1
    children = []
    for st in mod.structure_children:
        print("- number of atoms:", len(st.elements))
        if len(st.elements) > max_n_atom:
            continue
        enthalpy, st_opt, spacegroup = optimize_geometry(
            st,
            properties,
            gtol=gtol,
            maxiter=maxiter,
            pressure=pressure,
            verbose=verbose,
        )
        node_child = ModulationNode(st_opt, enthalpy, spacegroup, depth=depth)
        children.append(node_child)

    if len(children) == 0:
        print("All structures are composed of more than max_n_atom:", len(children))
        return node_parent

    # TODO: Remove duplicates using different sophisticated algorithm.
    children = eliminate_duplicate_structures(children)
    if depth > 6:
        print("Not converged.", len(children))
        os.makedirs("not_finished", exist_ok=True)
        for i, node_child in enumerate(children):
            filename = (
                "not_finished/poscar-"
                + node_child.spacegroup.replace("/", "").replace(" ", "")
                + "-"
                + str(i).zfill(3)
            )
            write_poscar_file(node_child.structure, filename=filename)
        return node_parent

    # if verbose:
    print("Number of unique modulated structures:", len(children))
    for node_child in children:
        sup = find_supercell_size_nearly_isotropic(node_child.structure)
        print(node_child.depth, sup)
        node_child = run_modulations(
            properties=properties,
            supercell_matrix=sup,
            node=node_child,
            max_denom=max_denom,
            max_n_atom=max_n_atom,
            gtol=gtol,
            maxiter=maxiter,
            pressure=pressure,
            depth=depth,
            verbose=verbose,
        )
        node_parent.add_child(node_child)
    return node_parent


# TODO: Implement function to save results of modulation calculations
