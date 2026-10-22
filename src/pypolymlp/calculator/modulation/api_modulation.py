"""API functions for modulation."""

# import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure

from .modulation import PolymlpModulation
from .modulation_utils import (
    ModulationNode,
    eliminate_duplicate_structures,
    find_supercell_size_nearly_isotropic,
    optimize_geometry,
)

# from pypolymlp.utils.vasp_utils import write_poscar_file


def run_modulations(
    properties: Properties,
    supercell_matrix: np.ndarray,
    unitcell: Optional[PolymlpStructure] = None,
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
        structure_children=mod.structure_children,
    )

    nodes = [[node_parent]]
    max_depth = 10
    while depth <= max_depth and len(nodes[depth]) > 0:
        depth += 1
        print("--- Depth:", depth, "---", flush=True)
        n_children = 0
        nodes_tmp = []
        for parent_id, node_parent in enumerate(nodes[depth - 1]):
            if len(node_parent.structure_children) == 0:
                print("Skipped: No imaginary")
                continue

            children = []
            print("- Parent:", node_parent.spacegroup, node_parent.structure.n_atoms)
            print(
                "Number of modulated structures:", len(node_parent.structure_children)
            )
            for st in node_parent.structure_children:
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
                node_child = ModulationNode(st_opt, enthalpy, spacegroup)
                children.append(node_child)

            if len(node_parent.structure_children) == 0:
                print("All structures are composed of more than max_n_atom.")
                continue

            # TODO: Remove duplicates using different sophisticated algorithm.
            children = eliminate_duplicate_structures(children)
            n_children += len(children)

            print("Number of unique modulated structures:", len(children))
            for node_child in children:
                sup = find_supercell_size_nearly_isotropic(node_child.structure)
                mod = PolymlpModulation(
                    properties=properties,
                    supercell_matrix=sup,
                    unitcell=node_child.structure,
                    pressure=pressure,
                    gtol=gtol,
                    maxiter=maxiter,
                    verbose=verbose,
                )
                mod.run_phonon_and_modulations(max_denom=max_denom, fc2=fc2)
                node_child_add = ModulationNode(
                    mod.unitcell,
                    mod.enthalpy,
                    mod.spacegroup,
                    is_imaginary=mod.is_imaginary,
                    depth=depth,
                    structure_children=mod.structure_children,
                    parent_id=parent_id,
                )
                print(node_child.spacegroup, list(sup), mod.is_imaginary)
                nodes_tmp.append(node_child_add)

        print("Node size:", len(nodes_tmp), n_children)
        nodes.append(nodes_tmp)

        if n_children == 0:
            break

    return nodes


# # TODO: Implement function to save results of modulation calculations
