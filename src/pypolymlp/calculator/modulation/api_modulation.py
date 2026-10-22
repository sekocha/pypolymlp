"""API functions for modulation."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure

from .modulation import PolymlpModulation
from .modulation_utils import (
    ModulationNode,
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
    fc2: Optional[np.ndarray] = None,
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
    print(mod.spacegroup)

    if unitcell is not None:
        node_parent = ModulationNode(
            mod.unitcell,
            mod.enthalpy,
            mod.spacegroup,
            is_imaginary=mod.is_imaginary,
        )
    else:
        node_parent = node
        node_parent.is_imaginary = mod.is_imaginary

    if not mod.is_imaginary:
        return node_parent

    # TODO: Remove duplicates.
    print("Number of modulated structures:", len(mod.structure_children))
    for st in mod.structure_children:
        enthalpy, unitcell, spacegroup = optimize_geometry(
            st,
            properties,
            gtol=gtol,
            maxiter=maxiter,
            pressure=pressure,
            verbose=verbose,
        )
        node_child = ModulationNode(unitcell, enthalpy, spacegroup)
        node_parent.add_child(node_child)

    # TODO: Supercell matrix must be automatically determined.
    for node_child in node_parent.children:
        sup = find_supercell_size_nearly_isotropic(node_child.structure)
        print(node_child.depth, sup)
        run_modulations(
            properties=properties,
            supercell_matrix=sup,
            node=node_child,
            max_denom=2,
            gtol=gtol,
            maxiter=maxiter,
            pressure=pressure,
            verbose=verbose,
        )
    return node_parent
