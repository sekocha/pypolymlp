"""Utility functions for restarting SSCHA."""

from typing import Literal, Optional, Union

import numpy as np
import yaml
from phono3py.file_IO import read_fc2_from_hdf5

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

        self._load_sscha_yaml()
        if fc2hdf5 is not None:
            self._fc2 = read_fc2_from_hdf5(fc2hdf5)

    def _load_sscha_yaml(self):
        """Load sscha_results.yaml file."""
        yaml_data = yaml.safe_load(open(self._yaml))

        if self._pot is None:
            self._pot = yaml_data["parameters"]["pot"]
        self._temp = yaml_data["parameters"]["temperature"]
        self._sscha_params = yaml_data["parameters"]

        properties = yaml_data["properties"]
        self._free_energy = properties["free_energy"]
        self._static_potential = properties["static_potential"]
        self._entropy = properties["entropy"]
        self._harmonic_heat_capacity = properties["harmonic_heat_capacity"]

        self._delta_fc = yaml_data["status"]["delta_fc"]
        self._converge = yaml_data["status"]["converge"]
        self._imaginary = yaml_data["status"]["imaginary"]
        self._logs = yaml_data["logs"]
        self._anharmonic_energy = self._logs["anharmonic_free_energy"][-1]

        self._unitcell = load_cell(yaml_data=yaml_data, tag="unitcell")
        self._supercell_matrix = np.array(yaml_data["supercell_matrix"])
        self._n_atom_unitcell = len(self._unitcell.elements)
        self._volume = np.linalg.det(self._unitcell.axis)

    @property
    def unit(self):
        """Return unit for free energy."""
        return self._unit

    @unit.setter
    def unit(self, unit_in: Literal["kJ/mol", "eV/cell", "eV/atom"]):
        """Set unit."""
        self._unit = unit_in

    @property
    def polymlp(self):
        """Return MLP file name."""
        return self._pot

    @property
    def temperature(self):
        """Return temperature."""
        return self._temp

    def _unit_conversion(self, val):
        """Convert unit for free energy and potentials."""
        if self._unit == "kJ/mol":
            return val
        elif self._unit == "eV/cell":
            return val / EVtoKJmol
        elif self._unit == "eV/atom":
            return val / EVtoKJmol / self._n_atom_unitcell
        raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")

    def _unit_conversion_entropy(self, val):
        """Convert unit for entropy and heat capacity."""
        if self._unit == "kJ/mol":
            return val
        elif self._unit == "eV/cell":
            return val / EVtoKJmol / 1000
        elif self._unit == "eV/atom":
            return val / EVtoKJmol / 1000 / self._n_atom_unitcell
        raise RuntimeError("Unit must be kJ/mol, eV/cell, or eV/atom")

    @property
    def free_energy(self):
        """Return free energy."""
        return self._unit_conversion(self._free_energy)

    @property
    def static_potential(self):
        """Return static potential energy."""
        return self._unit_conversion(self._static_potential)

    @property
    def entropy(self):
        """Return entropy."""
        return self._unit_conversion_entropy(self._entropy)

    @property
    def harmonic_heat_capacity(self):
        """Return harmonic heat capacity."""
        return self._unit_conversion_entropy(self._harmonic_heat_capacity)

    @property
    def anharmonic_energy(self):
        """Return static potential energy."""
        return self._unit_conversion(self._anharmonic_energy)

    @property
    def logs(self):
        """Return logs."""
        return self._logs

    @property
    def delta_fc(self):
        """Return FC difference between the last two iterations."""
        return self._delta_fc

    @property
    def converge(self):
        """Return convergence tag."""
        return self._converge

    @property
    def imaginary(self):
        """Return imaginary tag."""
        return self._imaginary

    @property
    def parameters(self):
        """Return simulation parameters."""
        return self._sscha_params

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
    def n_unitcells(self) -> int:
        """Return number of unit cells."""
        return int(round(np.linalg.det(self._supercell_matrix)))

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
    def volume(self):
        return self._volume
