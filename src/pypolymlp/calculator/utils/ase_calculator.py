"""ASE calculator class using pypolymlp."""

from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pypolymlp.calculator.properties import Properties, initialize_polymlp_calculator
from pypolymlp.calculator.utils.ase_utils import ase_atoms_to_structure
from pypolymlp.core.params import PolymlpParams

ALL_CHANGES = tuple(all_changes)


class PolymlpASECalculator(Calculator):
    """ASE calculator class using pypolymlp."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        pot: Optional[str, list[str]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        properties: Optional[Properties] = None,
        require_mlp: bool = True,
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
        self._prop = initialize_polymlp_calculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            return_none=not require_mlp,
        )
        self._use_reference = False
        self._use_fc2 = False

    def set_calculator(
        self,
        pot: Optional[str, list[str]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
    ):
        """Set polymlp."""
        self._prop = initialize_polymlp_calculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
        )

    def calculate(
        self,
        atoms: Atoms,
        properties: tuple = ("energy", "forces", "stress"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`."""
        super().calculate(atoms, properties, system_changes)
        structure = ase_atoms_to_structure(atoms)
        energy, forces, stress = self._prop.eval(structure)
        self.results["energy"] = energy
        self.results["forces"] = forces.T
        self.results["stress"] = stress
