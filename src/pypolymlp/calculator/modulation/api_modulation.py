"""API functions for modulation."""

# import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure

from .modulation import PolymlpModulation
from .modulation_io import save_nodes_to_file
from .modulation_utils import (
    eliminate_duplicate_structures,
    exist_in_former_nodes,
    find_supercell_size_nearly_isotropic,
)


def run_modulations(
    properties: Properties,
    supercell_matrix: np.ndarray,
    unitcell: Optional[PolymlpStructure] = None,
    gtol: float = 1e-4,
    maxiter: int = 10000,
    pressure: float = 0.0,
    max_denom: int = 3,
    max_n_atom: int = 64,
    max_depth: int = 10,
    depth: int = 0,
    verbose: bool = False,
    verbose_go: bool = False,
):
    """Run recursively calculations for modulated structures."""
    mod = PolymlpModulation(
        properties=properties, pressure=pressure, verbose=verbose, verbose_go=verbose_go
    )
    node_parent = mod.run_phonon_and_modulations(
        unitcell,
        supercell_matrix,
        gtol=gtol,
        maxiter=maxiter,
        max_denom=max_denom,
    )
    node_parent.depth = depth

    nodes_flat = [node_parent]
    nodes = [[node_parent]]
    while depth <= max_depth and len(nodes[depth]) > 0:
        depth += 1
        if verbose:
            print("--- Depth:", depth, "---", flush=True)

        nodes_children = []
        for parent_id, node_parent in enumerate(nodes[depth - 1]):
            if len(node_parent.structures_children) == 0:
                continue
            if node_parent.equivalent_shallower is not None:
                continue

            if verbose:
                print(
                    "- Parent:",
                    node_parent.spacegroup,
                    node_parent.structure.n_atoms,
                    node_parent.enthalpy,
                    flush=True,
                )

            children_tmp = mod.optimize_geometry_multiple(
                node_parent.structures_children,
                gtol=gtol,
                maxiter=maxiter,
                max_n_atom=max_n_atom,
            )
            if len(children_tmp) == 0:
                if verbose:
                    print(
                        "All structures are composed of more than max_n_atom.",
                        flush=True,
                    )
                continue

            # TODO: Remove duplicates using different sophisticated algorithm.
            children_tmp = eliminate_duplicate_structures(children_tmp)

            child_id = 0
            for node_child in children_tmp:
                sup = find_supercell_size_nearly_isotropic(node_child.structure)
                sup = np.maximum(sup, [2, 2, 2])
                node_child_add = mod.run_phonon_and_modulations(
                    node_child.structure,
                    sup,
                    gtol=gtol,
                    maxiter=maxiter,
                    max_denom=max_denom,
                )
                node_child_add.depth = depth
                node_child_add.parent_id = parent_id
                node_child_add.id = child_id
                node_child_add = exist_in_former_nodes(node_child_add, nodes_flat)

                nodes_children.append(node_child_add)
                if node_child_add.equivalent_shallower is None:
                    nodes_flat.append(node_child_add)
                child_id += 1

                if verbose:
                    print(
                        node_child_add.spacegroup,
                        node_child_add.structure.n_atoms,
                        list(sup),
                        node_child_add.is_imaginary,
                        node_child_add.enthalpy,
                        flush=True,
                    )

        nodes.append(nodes_children)
        if len(nodes_children) == 0:
            break

    return nodes


def save_nodes(nodes: list, filename: str = "polymlp_modulation.yaml"):
    """Save attributes of nodes."""
    save_nodes_to_file(nodes, filename=filename)
