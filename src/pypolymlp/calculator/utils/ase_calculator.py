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


def convert_atoms_to_str(atoms: Atoms, structure_without_disp: PolymlpStructure):
    """Convert ASE atoms to structure and displacement."""
    structure = ase_atoms_to_structure(atoms)
    disps = convert_positions_to_disps(structure, structure_without_disp)
    disps = disps.T.reshape(-1)
    return disps, structure


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

        self._use_reference = True
        self._use_fc2 = True
        self._delta_energy_10 = None
        self._delta_energy_1a = None
        self._average_displacement = None

    def _check_errors(self):
        """Check errors in input parameters."""
        n_atom = self._structure_without_disp.positions.shape[1]
        assert self._fc2.shape[0] == n_atom * 3
        assert self._fc2.shape[1] == n_atom * 3
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
        """Calculate energy, force, and stress using `pypolymlp`.

        energy: E(alpha).
        delta_energy_10: E - E_ref. Its average is <E - E_ref>_alpha.
        delta_energy_1a: E - E(alpha). Its average is <E - E(alpha)>_alpha.
        """
        super().calculate(atoms, properties, system_changes)
        disps, structure = convert_atoms_to_str(atoms, self._structure_without_disp)
        self._average_displacement = np.sqrt(np.average(np.square(disps)))

        energy0, forces0 = self._eval_fc2_model(disps)
        energy1, forces1, _ = self._prop.eval(structure)
        energy = energy1 * self._alpha + energy0 * (1 - self._alpha)
        forces = forces1 * self._alpha + forces0 * (1 - self._alpha)
        self._delta_energy_10 = energy1 - energy0
        self._delta_energy_1a = energy1 - energy

        self.results["energy"] = energy
        self.results["forces"] = forces.T

    @property
    def delta_energy_10(self):
        """Return energy difference from reference state.

        delta_energy_10: E - E_ref. Its average is <E - E_ref>_alpha.
        """
        return self._delta_energy_10

    @property
    def delta_energy_1a(self):
        """Return energy difference from state at alpha.

        delta_energy_1a: E - E_alpha. Its average is <E - E_alpha>_alpha.
        """
        return self._delta_energy_1a

    @property
    def average_displacement(self):
        """Return average displacement."""
        return self._average_displacement

    @property
    def alpha(self):
        """Return alpha."""
        return self._alpha

    @property
    def static_energy(self):
        """Return potential energy of structure with displacements."""
        return self._static_energy

    @alpha.setter
    def alpha(self, _alpha):
        """Set alpha."""
        self._alpha = _alpha


class PolymlpRefASECalculator(Calculator):
    """ASE calculator class using difference between two pypolymlps."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        pot: Optional[Union[str, list]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        pot_ref: Optional[Union[str, list]] = None,
        params_ref: Optional[PolymlpParams] = None,
        coeffs_ref: Optional[np.ndarray] = None,
        properties_ref: Optional[Properties] = None,
        alpha: float = 0.0,
        **kwargs,
    ):
        """Initialize PolymlpRefASECalculator.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.
        pot_ref: polymlp file for reference state.
        params_ref: Parameters for polymlp for reference state.
        coeffs_ref: Polymlp coefficients for reference state.
        properties_ref: Properties object for reference state.
        alpha: Mixing parameter. E = alpha * E_polymlp + (1 - alpha) * E_polymlp_ref
        """
        super().__init__(**kwargs)
        self._prop = set_instance_properties(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            require_mlp=True,
        )
        self._prop_ref = set_instance_properties(
            pot=pot_ref,
            params=params_ref,
            coeffs=coeffs_ref,
            properties=properties_ref,
            require_mlp=True,
        )
        self._alpha = alpha
        self._check_errors()

        self._use_reference = True
        self._use_fc2 = False
        self._delta_energy_10 = None
        self._delta_energy_1a = None

    def _check_errors(self):
        """Check errors in input parameters."""
        assert self._alpha >= 0.0
        assert self._alpha <= 1.0

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: tuple = ("energy", "forces"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`.

        energy: E(alpha).
        delta_energy_10: E - E_ref. Its average is <E - E_ref>_alpha.
        delta_energy_1a: E - E(alpha). Its average is <E - E(alpha)>_alpha.
        """
        super().calculate(atoms, properties, system_changes)
        structure = ase_atoms_to_structure(atoms)

        energy0, forces0, stress0 = self._prop_ref.eval(structure)
        energy1, forces1, stress1 = self._prop.eval(structure)
        energy = energy1 * self._alpha + energy0 * (1 - self._alpha)
        forces = forces1 * self._alpha + forces0 * (1 - self._alpha)
        stress = stress1 * self._alpha + stress0 * (1 - self._alpha)
        self._delta_energy_10 = energy1 - energy0
        self._delta_energy_1a = energy1 - energy

        self.results["energy"] = energy
        self.results["forces"] = forces.T
        self.results["stress"] = stress

    @property
    def delta_energy_10(self):
        """Return energy difference from reference state.

        delta_energy_10: E - E_ref. Its average is <E - E_ref>_alpha.
        """
        return self._delta_energy_10

    @property
    def delta_energy_1a(self):
        """Return energy difference from state at alpha.

        delta_energy_1a: E - E(alpha). Its average is <E - E(alpha)>_alpha.
        """
        return self._delta_energy_1a

    @property
    def alpha(self):
        """Return alpha."""
        return self._alpha

    @alpha.setter
    def alpha(self, _alpha):
        """Set alpha."""
        self._alpha = _alpha


class PolymlpGeneralRefASECalculator(Calculator):
    """ASE calculator class using difference between general two states."""

    implemented_properties = ("energy", "forces", "stress")

    def __init__(
        self,
        fc2: np.ndarray,
        structure_without_disp: PolymlpStructure,
        pot_final: Optional[Union[str, list]] = None,
        params_final: Optional[PolymlpParams] = None,
        coeffs_final: Optional[np.ndarray] = None,
        properties_final: Optional[Properties] = None,
        pot_ref: Optional[Union[str, list]] = None,
        params_ref: Optional[PolymlpParams] = None,
        coeffs_ref: Optional[np.ndarray] = None,
        properties_ref: Optional[Properties] = None,
        alpha_final: float = 0.0,
        alpha_ref: float = 0.0,
        alpha: float = 0.0,
        **kwargs,
    ):
        """Initialize PolymlpGeneralRefASECalculator.

        Parameters
        ----------
        pot_final: polymlp file for final state.
        params_final: Parameters for polymlp for final state.
        coeffs_final: Polymlp coefficients for final state.
        properties_final: Properties object for final state.

        pot_ref: polymlp file for reference state.
        params_ref: Parameters for polymlp for reference state.
        coeffs_ref: Polymlp coefficients for reference state.
        properties_ref: Properties object for reference state.

        alpha_ref: Mixing parameter for defining reference state.
            E = alpha * E_polymlp_ref + (1 - alpha) * E_fc2
        alpha_final: Mixing parameter for defining final state.
            E = alpha * E_polymlp_final + (1 - alpha) * E_fc2
        alpha: Mixing parameter.
            E = alpha * E_final + (1 - alpha) * E_ref
        """
        super().__init__(**kwargs)
        self._prop_final = set_instance_properties(
            pot=pot_final,
            params=params_final,
            coeffs=coeffs_final,
            properties=properties_final,
            require_mlp=True,
        )
        self._prop_ref = set_instance_properties(
            pot=pot_ref,
            params=params_ref,
            coeffs=coeffs_ref,
            properties=properties_ref,
            require_mlp=True,
        )
        self._alpha_final = alpha_final
        self._alpha_ref = alpha_ref
        self._alpha = alpha
        self._fc2 = fc2
        self._structure_without_disp = structure_without_disp
        self._check_errors()

        self._static_energy_ref, _, _ = self._prop_ref.eval(structure_without_disp)
        self._static_energy_final, _, _ = self._prop_final.eval(structure_without_disp)

        self._use_reference = True
        self._use_fc2 = True
        self._delta_energy_10 = None
        self._delta_energy_1a = None
        self._average_displacement = None

    def _check_errors(self):
        """Check errors in input parameters."""
        n_atom = self._structure_without_disp.positions.shape[1]
        assert self._fc2.shape[0] == n_atom * 3
        assert self._fc2.shape[1] == n_atom * 3
        assert self._alpha_final >= 0.0
        assert self._alpha_final <= 1.0
        assert self._alpha_ref >= 0.0
        assert self._alpha_ref <= 1.0
        assert self._alpha >= 0.0
        assert self._alpha <= 1.0

    def _eval_linear_model(
        self,
        disps: np.array,
        structure: PolymlpStructure,
        static_energy: float,
        prop: Properties,
        fc2: np.ndarray,
        alpha: float,
    ):
        """Calculate energy and forces of a linear combination of polymlp and FC2."""
        energy0, forces0 = eval_properties_fc2(fc2, disps)
        energy0 += static_energy
        energy1, forces1, _ = prop.eval(structure)

        energy = energy1 * alpha + energy0 * (1 - alpha)
        forces = forces1 * alpha + forces0 * (1 - alpha)
        return energy, forces

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: tuple = ("energy", "forces"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`.

        energy: E(alpha).
        delta_energy_10: E - E_ref. Its average is <E - E_ref>_alpha.
        delta_energy_1a: E - E(alpha). Its average is <E - E(alpha)>_alpha.
        """
        super().calculate(atoms, properties, system_changes)
        disps, structure = convert_atoms_to_str(atoms, self._structure_without_disp)
        energy0, forces0 = self._eval_linear_model(
            disps,
            structure,
            self._static_energy_ref,
            self._prop_ref,
            self._fc2,
            self._alpha_ref,
        )
        energy1, forces1 = self._eval_linear_model(
            disps,
            structure,
            self._static_energy_final,
            self._prop_final,
            self._fc2,
            self._alpha_final,
        )
        energy = energy1 * self._alpha + energy0 * (1 - self._alpha)
        forces = forces1 * self._alpha + forces0 * (1 - self._alpha)

        self._average_displacement = np.sqrt(np.average(np.square(disps)))
        self._delta_energy_10 = energy1 - energy0
        self._delta_energy_1a = energy1 - energy

        self.results["energy"] = energy
        self.results["forces"] = forces.T

    @property
    def delta_energy_10(self):
        """Return energy difference from reference state.

        delta_energy_10: E - E_ref. Its average is <E - E_ref>_alpha.
        """
        return self._delta_energy_10

    @property
    def delta_energy_1a(self):
        """Return energy difference from state at alpha.

        delta_energy_1a: E - E(alpha). Its average is <E - E(alpha)>_alpha.
        """
        return self._delta_energy_1a

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
        """Set alpha."""
        self._alpha = _alpha
