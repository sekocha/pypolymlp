"""Utility functions for modulation."""

from dataclasses import dataclass
from typing import Optional, Self

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

    parent: Optional[Self] = None
    children: Optional[list] = None
    depth: int = 0

    def __post_init__(self):
        """Init method."""
        self.children = []

    def add_child(self, child):
        """Add child."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def traverse(self):
        yield self
        for child in self.children:
            yield from child.traverse()


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
    unitcell = sym.refine_cell()
    spacegroup = sym.get_spacegroup()
    return (enthalpy, unitcell, spacegroup)
