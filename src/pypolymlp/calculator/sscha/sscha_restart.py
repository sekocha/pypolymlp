"""Utility functions for restarting SSCHA."""

from typing import Literal, Optional, Union

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5

from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.phonopy_utils import phonopy_supercell, structure_to_phonopy_cell
from pypolymlp.utils.yaml_utils import load_cell


class Restart:
    """Class for reading files to restart SSCHA."""

    def __init__(
        self,
        res_yaml: str,
        fc2hdf5: Optional[str] = None,
        pot: Optional[Union[str, list, tuple]] = None,
        unit: Literal["kJ/mol", "eV/cell", "eV/atom"] = "kJ/mol",
    ):
        """Init method."""
        self._yaml = res_yaml
        self._pot = pot
        self._unit = unit

        self._unitcell = None
        self._supercell_matrix = None
        self._sscha_params = None
        self._sscha_data = None
        self._sscha_status = None
        self._sscha_logs = None
        self._fc2 = None

        self._load_sscha_yaml()
        self._set_unit_conversion()

        if fc2hdf5 is not None:
            self._fc2 = read_fc2_from_hdf5(fc2hdf5)

    def _load_sscha_yaml(self):
        """Load sscha_results.yaml file."""
        yaml_data = yaml.safe_load(open(self._yaml))
        if self._pot is None:
            self._pot = yaml_data["parameters"]["pot"]

        self._sscha_params = yaml_data["parameters"]
        properties = yaml_data["properties"]
        self._sscha_status = yaml_data["status"]
        self._sscha_logs = yaml_data["logs"]
        self._sscha_data = SSCHAData(
            temperature=self._sscha_params["temperature"],
            static_potential=properties["static_potential"],  # kJ/mol
            harmonic_free_energy=properties["harmonic_free_energy"],  # kJ/mol
            anharmonic_free_energy=properties["anharmonic_free_energy"],  # kJ/mol
            free_energy=properties["free_energy"],  # kJ/mol
            entropy=properties["entropy"],  # J/K/mol
            harmonic_heat_capacity=properties["harmonic_heat_capacity"],  # J/K/mol
        )

        self._unitcell = load_cell(yaml_data=yaml_data, tag="unitcell")
        self._supercell_matrix = np.array(yaml_data["supercell_matrix"])
        self._n_atom_unitcell = len(self._unitcell.elements)
        self._volume = np.linalg.det(self._unitcell.axis)
        return self

    @property
    def unit(self):
        """Return unit for free energy."""
        return self._unit

    @unit.setter
    def unit(self, unit_in: Literal["kJ/mol", "eV/cell", "eV/atom"]):
        """Set unit."""
        self._unit = unit_in
        self._unit_energy, self._unit_entropy = self._set_unit_conversion()

    def _set_unit_conversion(self):
        """Set unit conversion values."""
        if self._unit not in ("kJ/mol", "eV/cell", "eV/atom"):
            raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")

        if self._unit == "kJ/mol":
            self._unit_energy = 1.0
            self._unit_entropy = 1.0
            self._unit_volume = 1.0
        elif self._unit == "eV/cell":
            self._unit_energy = 1.0 / EVtoKJmol
            self._unit_entropy = 1.0 / (EVtoKJmol * 1000)
            self._unit_volume = 1.0
        elif self._unit == "eV/atom":
            self._unit_energy = 1.0 / (EVtoKJmol * self._n_atom_unitcell)
            self._unit_entropy = 1.0 / (EVtoKJmol * 1000 * self._n_atom_unitcell)
            self._unit_volume = 1.0 / self._n_atom_unitcell
        return self

    @property
    def polymlp(self):
        """Return MLP file name."""
        return self._pot

    @property
    def temperature(self):
        """Return temperature."""
        return self._sscha_data.temperature

    @property
    def free_energy(self):
        """Return free energy."""
        return self._sscha_data.free_energy * self._unit_energy

    @property
    def static_potential(self):
        """Return static potential energy."""
        return self._sscha_data.static_potential * self._unit_energy

    @property
    def harmonic_free_energy(self):
        """Return harmonic free energy."""
        return self._sscha_data.harmonic_free_energy * self._unit_energy

    @property
    def anharmonic_free_energy(self):
        """Return anharmonic free energy."""
        return self._sscha_data.anharmonic_free_energy * self._unit_energy

    @property
    def entropy(self):
        """Return entropy."""
        return self._sscha_data.entropy * self._unit_entropy

    @property
    def harmonic_heat_capacity(self):
        """Return harmonic heat capacity."""
        return self._sscha_data.harmonic_heat_capacity * self._unit_entropy

    @property
    def parameters(self):
        """Return simulation parameters."""
        return self._sscha_params

    @property
    def logs(self):
        """Return logs."""
        return self._sscha_logs

    @property
    def delta_fc(self):
        """Return FC difference between the last two iterations."""
        return self._sscha_status["delta_fc"]

    @property
    def converge(self):
        """Return convergence tag."""
        return self._sscha_status["converge"]

    @property
    def imaginary(self):
        """Return imaginary tag."""
        return self._sscha_status["imaginary"]

    @property
    def force_constants(self):
        """Return effective FC2."""
        return self._fc2

    @property
    def unitcell(self) -> PolymlpStructure:
        """Return unitcell."""
        return self._unitcell

    @property
    def unitcell_phonopy(self):
        """Return unitcell in PhonopyAtoms."""
        return structure_to_phonopy_cell(self._unitcell)

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return supercell matrix."""
        return self._supercell_matrix

    @property
    def supercell(self) -> PolymlpStructure:
        """Return supercell."""
        cell = phonopy_supercell(
            self._unitcell,
            supercell_matrix=self._supercell_matrix,
            return_phonopy=False,
        )
        return cell

    @property
    def supercell_phonopy(self):
        """Return supercell in PhonopyAtoms."""
        cell = phonopy_supercell(
            self._unitcell,
            supercell_matrix=self._supercell_matrix,
            return_phonopy=True,
        )
        return cell

    @property
    def n_unitcells(self) -> int:
        """Return number of unit cells."""
        return int(round(np.linalg.det(self._supercell_matrix)))

    @property
    def volume(self):
        """Return volume."""
        return self._volume * self._unit_volume

    # def _unit_conversion(self, val):
    #     """Convert unit for free energy and potentials."""
    #     if self._unit == "kJ/mol":
    #         return val
    #     elif self._unit == "eV/cell":
    #         return val / EVtoKJmol
    #     elif self._unit == "eV/atom":
    #         return val / EVtoKJmol / self._n_atom_unitcell
    #     raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")

    # def _unit_conversion_entropy(self, val):
    #     """Convert unit for entropy and heat capacity."""
    #     if self._unit == "kJ/mol":
    #         return val
    #     elif self._unit == "eV/cell":
    #         return val / EVtoKJmol / 1000
    #     elif self._unit == "eV/atom":
    #         return val / EVtoKJmol / 1000 / self._n_atom_unitcell
    #     raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")
