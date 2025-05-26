"""ASE calculator class using pypolymlp."""

from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pypolymlp.calculator.properties import Properties, set_instance_properties
from pypolymlp.calculator.utils.ase_utils import ase_atoms_to_structure
from pypolymlp.calculator.utils.fc_utils import eval_properties_fc2
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.displacements import convert_positions_to_disps

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
        self._prop = set_instance_properties(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            require_mlp=require_mlp,
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


class PolymlpFC2ASECalculator(Calculator):
    """ASE calculator class using difference between pypolymlp and fc2."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        fc2: np.ndarray,
        structure_without_disp: PolymlpStructure,
        pot: Optional[Union[str, list]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        alpha: float = 0.0,
        **kwargs,
    ):
        """Initialize PolymlpFC2ASECalculator.

        Parameters
        ----------
        fc2: Second-order force constants. shape=(N3, N3).
        structure_without_disp:
            Structure where displacements are not included in PolymlpStructure format.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.
        alpha: Mixing parameter. E = alpha * E_polymlp + (1 - alpha) * E_fc2
        """
        super().__init__(**kwargs)
        self._prop = set_instance_properties(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
        )
        self._fc2 = fc2
        self._alpha = alpha
        self._structure_without_disp = structure_without_disp
        self._check_errors()

        self._static_energy, _, _ = self._prop.eval(structure_without_disp)
        self._ignore_polymlp = np.isclose(alpha, 0.0)

        self._use_reference = True
        self._delta_energy = None
        self._average_displacement = None

    def _check_errors(self):
        """Check errors in input parameters."""
        assert self._fc2.shape[0] == self._structure_without_disp.positions.shape[1] * 3
        assert self._fc2.shape[1] == self._structure_without_disp.positions.shape[1] * 3
        assert self._alpha >= 0.0
        assert self._alpha <= 1.0

    def _eval_fc2_model(self, disps: np.array):
        """Calculate energy and forces using FC2."""
        energy, forces = eval_properties_fc2(self._fc2, disps)
        energy += self._static_energy
        return energy, forces

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: tuple = ("energy", "forces"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`."""
        super().calculate(atoms, properties, system_changes)
        structure = ase_atoms_to_structure(atoms)
        disps = convert_positions_to_disps(structure, self._structure_without_disp)
        disps = disps.T.reshape(-1)
        self._average_displacement = np.sqrt(np.average(np.square(disps)))

        if self._ignore_polymlp:
            energy, forces = self._eval_fc2_model(disps)
            self._delta_energy = 0.0
        else:
            energy1, forces1 = self._eval_fc2_model(disps)
            energy2, forces2, _ = self._prop.eval(structure)
            self._delta_energy = energy2 - energy1

            energy = energy2 * self._alpha + energy1 * (1 - self._alpha)
            forces = forces2 * self._alpha + forces1 * (1 - self._alpha)

        self.results["energy"] = energy
        self.results["forces"] = forces.T

    @property
    def delta_energy(self):
        """Return energy difference from reference state."""
        return self._delta_energy

    @property
    def average_displacement(self):
        """Return average displacement."""
        return self._average_displacement

    @property
    def alpha(self):
        """Return alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, _alpha):
        """Return alpha."""
        self._alpha = _alpha
        self._ignore_polymlp = np.isclose(_alpha, 0.0)
