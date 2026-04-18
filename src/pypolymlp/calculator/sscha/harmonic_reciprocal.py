"""Class for harmonic contribution in reciprocal space."""

from typing import Optional

import numpy as np
from phonopy import Phonopy

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure


class HarmonicReciprocal:
    """Class for harmonic contribution in reciprocal space."""

    def __init__(
        self,
        phonopy_obj: Phonopy,
        properties: Properties,
        fc2: Optional[np.ndarray] = None,
    ):
        """Init method.

        Parameters
        ----------
        phonopy_obj: Phonopy class object.
        property: Property class object to calculate energies and forces.
        fc2: Second-order force constants.
        """

        self._ph = phonopy_obj
        self._n_atom = len(self._ph.supercell.numbers)

        self._prop = properties
        self.force_constants = fc2

        self._tp_dict = dict()
        self._mesh_dict = dict()

        self._frequencies = None
        self._is_imaginary = None

    def eval(self, structures: list[PolymlpStructure]):
        """Compute energies and forces of structures.

        Parameters
        ----------
        structures: Structures.

        Return
        ------
        energies: Energies, shape=(n_str)
        forces: Forces, shape=(n_str, 3, n_atom)
        """
        energies, forces, _ = self._prop.eval_multiple(structures)
        return np.array(energies), np.array(forces)

    def produce_harmonic_force_constants(self, displacements: float = 0.01):
        """Produce non-effective harmonic FCs."""
        self._ph.generate_displacements(distance=displacements)
        supercells = self._ph.supercells_with_displacements
        _, forces = self.eval([phonopy_cell_to_structure(cell) for cell in supercells])
        self._ph.forces = np.array(forces).transpose((0, 2, 1))
        self._ph.produce_force_constants()
        self._fc2 = self._ph.force_constants
        return self._fc2

    def _hide_imaginary_modes(self, freq: np.ndarray, freq_threshold: float = -0.001):
        """Mask branches with imaginary frequencies."""
        freq_rev = np.array(freq)
        match1 = np.where(freq_rev < freq_threshold)
        freq_rev[match1] = np.abs(freq_rev[match1]) * 10.0
        return freq_rev

    def compute_mesh_properties(self, qmesh: tuple = (10, 10, 10)):
        """Compute mesh properties."""
        self._ph.run_mesh(qmesh)
        self._frequencies = self._ph._mesh.frequencies

        self._ph.run_total_dos()
        self._mesh_dict = self._ph.get_mesh_dict()
        return self

    def compute_thermal_properties(
        self,
        temp: float = 1000,
        qmesh: tuple = (10, 10, 10),
        hide_imaginary: bool = False,
    ):
        """Compute thermal properties."""
        self._ph.run_mesh(qmesh)
        self._frequencies = self._ph._mesh.frequencies
        self._ph.run_thermal_properties(t_step=10, t_max=temp, t_min=temp)
        self._tp_dict = self._ph.get_thermal_properties_dict()
        free_energy_raw = self._tp_dict["free_energy"][0]

        free_energy_rev = None
        if hide_imaginary:
            freq = self._ph._mesh._frequencies
            self._ph._mesh._frequencies = self._hide_imaginary_modes(freq)
            self._ph.run_thermal_properties(t_step=10, t_max=temp, t_min=temp)
            self._tp_dict = self._ph.get_thermal_properties_dict()
            free_energy_rev = self._tp_dict["free_energy"][0]
        return free_energy_raw, free_energy_rev

    @property
    def force_constants(self):
        """Return FC2, shape=(n_atom, n_atom, 3, 3)."""
        return self._fc2

    @force_constants.setter
    def force_constants(self, fc2: np.ndarray):
        """Set FC2, shape=(n_atom, n_atom, 3, 3)."""
        if fc2 is None:
            self._fc2 = None
            return
        assert fc2.shape[0] == fc2.shape[1] == self._n_atom
        assert fc2.shape[2] == fc2.shape[3] == 3
        self._fc2 = fc2
        self._ph.force_constants = fc2

    @property
    def free_energy(self):
        """Return harmonic free energy."""
        try:
            return self._tp_dict["free_energy"][0]
        except:
            return None

    @property
    def entropy(self):
        """Return entropy."""
        try:
            return self._tp_dict["entropy"][0]
        except:
            return None

    @property
    def heat_capacity(self):
        """Return heat capacity."""
        try:
            return self._tp_dict["heat_capacity"][0]
        except:
            return None

    @property
    def frequencies(self):
        """Return phonon frequencies."""
        try:
            return self._mesh_dict["frequencies"]
        except:
            return None

    @property
    def phonopy_instance(self):
        """Return phonopy instance."""
        return self._ph

    @property
    def mesh_dict(self):
        """Return mesh dict."""
        return self._mesh_dict

    @property
    def tp_dict(self):
        """Return thermodynamic properties in dict."""
        return self._tp_dict

    @property
    def is_imaginary(self):
        """Return whether frequencies show imaginary."""
        if self._frequencies is None:
            raise RuntimeError("Frequencies not found.")

        tol = -0.01
        n_imag = np.count_nonzero(self._frequencies < tol)
        return (n_imag / self._frequencies.size) > 1e-5
