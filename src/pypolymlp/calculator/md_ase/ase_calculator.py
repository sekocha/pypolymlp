"""ASE calculator class using pypolymlp."""

from typing import Optional, Union

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pypolymlp.calculator.properties import Properties
from pypolymlp.utils.ase_utils import ase_atoms_to_structure

ALL_CHANGES = tuple(all_changes)


class PolymlpASECalculator(Calculator):
    """ASE calculator class using pypolymlp."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, potentials: Optional[Union[str, list]] = None, **kwargs):
        """Initialize PolymlpASECalculator."""
        super().__init__(**kwargs)
        self._prop = None
        if potentials is not None:
            self.set_calculator(potentials)

    def set_calculator(self, potentials: Union[str, list]):
        """Set polymlp."""
        self._prop = Properties(pot=potentials)

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: tuple = ("energy", "forces"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`."""
        super().calculate(atoms, properties, system_changes)
        structure = ase_atoms_to_structure(atoms)
        energy, forces, stress = self._properties.eval(structure)
        self.results["energy"] = energy
        self.results["forces"] = forces.T
        self.results["stress"] = stress
