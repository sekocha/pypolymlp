"""ASE calculator class using pypolymlp."""

from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pypolymlp.calculator.properties import Properties, set_instance_properties
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.utils.ase_utils import ase_atoms_to_structure

ALL_CHANGES = tuple(all_changes)


class PolymlpASECalculator(Calculator):
    """ASE calculator class using pypolymlp."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        pot: Optional[Union[str, list]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        **kwargs,
    ):
        """Initialize PolymlpASECalculator.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.
        """
        super().__init__(**kwargs)
        self._prop = set_instance_properties(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
        )

    def set_calculator(
        self,
        pot: Optional[Union[str, list]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
    ):
        """Set polymlp."""
        self._prop = set_instance_properties(pot=pot, params=params, coeffs=coeffs)

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: tuple = ("energy", "forces"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`."""
        super().calculate(atoms, properties, system_changes)
        structure = ase_atoms_to_structure(atoms)
        energy, forces, stress = self._prop.eval(structure)
        self.results["energy"] = energy
        self.results["forces"] = forces.T
        self.results["stress"] = stress
