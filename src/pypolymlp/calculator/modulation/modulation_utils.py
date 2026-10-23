"""Utility functions for modulation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.spglib_utils import SymCell


@dataclass
class ModulationNode:
    """Dataclass for modulation tree node."""

    structure: PolymlpStructure
    enthalpy: float
    spacegroup: Optional[str] = None
    is_imaginary: bool = False

    structures_children: Optional[list] = None
    depth: int = 0
    parent_id: int = 0
    id: int = 0
    equivalent_shallower: Optional[tuple] = None


def optimize_geometry(
    st: PolymlpStructure,
    prop: Properties,
    gtol: float = 1e-4,
    maxiter: int = 10000,
    pressure: float = 0.0,
    verbose: bool = False,
):
    go = GeometryOptimization(
        st,
        prop,
        with_sym=True,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        pressure=pressure,
        verbose=verbose,
    )
    go.run(gtol=gtol, maxiter=maxiter)
    enthalpy = go.enthalpy / len(go.structure.elements)

    sym = SymCell(st=go.structure, symprec=1e-4)
    spacegroup = sym.get_spacegroup()
    return (enthalpy, go.structure, spacegroup)


def find_supercell_size_nearly_isotropic(
    structure: PolymlpStructure,
    natom_lb: int = 48,
    natom_ub: int = 150,
):
    """Find a diagonal supercell size enabling nearly-isotropic supercell."""
    axis = structure.axis
    total_n_atoms = sum(structure.n_atoms)

    len_axis = [np.linalg.norm(axis[:, i]) for i in range(3)]
    ratio1 = len_axis[0] / len_axis[1]
    ratio2 = len_axis[0] / len_axis[2]
    ratio = np.array([1, ratio1, ratio2])

    cand = np.arange(1, 11)
    size = [1, 1, 1]
    for c in cand:
        size_trial = np.maximum(np.round(ratio * c).astype(int), [1, 1, 1])
        n_total = total_n_atoms * np.prod(size_trial)
        if n_total >= natom_lb:
            if n_total <= natom_ub:
                size = size_trial
            break
        size = size_trial
    return size


def eliminate_duplicate_structures(
    nodes: list[ModulationNode],
    threshold: float = 1e-5,
):
    """Eliminate duplicate structures."""
    enthalpies = np.array([n.enthalpy for n in nodes])
    rounded = np.round(enthalpies / threshold) * threshold
    _, unique_indices = np.unique(rounded, return_index=True)
    return [nodes[i] for i in np.sort(unique_indices)]


def exist_in_former_nodes(
    node: ModulationNode,
    nodes_flat: list[ModulationNode],
    threshold: float = 1e-5,
):
    """Check whether a node exists in former nodes."""
    enthalpies = np.array([n.enthalpy for n in nodes_flat])
    rounded_ref = np.round(enthalpies / threshold) * threshold
    rounded = np.round(node.enthalpy / threshold) * threshold
    match = np.where(rounded_ref == rounded)[0]
    if len(match) == 0:
        return node

    for m in match:
        node_ref = nodes_flat[m]
        if node.spacegroup == node_ref.spacegroup:
            key = (node_ref.depth, node_ref.parent_id, node_ref.id)
            print(key)
            node.equivalent_shallower = key
            break
    return node
