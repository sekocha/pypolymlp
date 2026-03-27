"""Class for defining compositions."""

import numpy as np


class Composition:
    """Class for defining compositions."""

    def __init__(self, chemical_comps_end_members: np.ndarray):
        """Init method.

        Parameters
        ----------
        chemical_comps_end_members: Chemical compositions for end members.
            shape=(n_end_members, n_type),
            Each row corresponds to number of atoms for each end member.
            For example, if SnO and SnO2 are endmembers that are used
            to define the composition, this array should be given as
            chemical_comps_end_members = [[1, 1], [1, 2]].
        """
        self._composition_axis = np.array(chemical_comps_end_members).T
        self._composition_axis_inv = np.linalg.pinv(self._composition_axis)
        self._proj = self._composition_axis @ self._composition_axis_inv

        self._energies_end = None

        self._comp = None

    def get_composition(self, n_atoms: np.ndarray):
        """Return composition.

        Parameter
        ---------
        n_atoms: Number of atoms in structure, shape = (n_type).
        """
        if not np.allclose(self._proj @ n_atoms, n_atoms):
            raise RuntimeError("The composition not in the given compositional space.")

        partition = self._composition_axis_inv @ n_atoms
        self._comp = partition / np.sum(partition)
        return self._comp, partition

    def get_compositions(self, n_atoms_array: np.ndarray):
        """Return compositions.

        Parameter
        ---------
        n_atoms_array: Numbers of atoms in structures.
                       shape = (n_str, n_type)
        """
        n_atoms = np.array(n_atoms_array).T
        if not np.allclose(self._proj @ n_atoms, n_atoms):
            raise RuntimeError("The composition not in the given compositional space.")

        partition = self._composition_axis_inv @ n_atoms
        partition = partition.T
        self._comp = partition / np.sum(partition, axis=1)[:, None]
        return self._comp, partition

    def compute_formation_energy(self, energy: float, n_atoms: np.ndarray):
        """Compute formation energy.

        Energy per n_atoms must be given.
        """
        if self._energies_end is None:
            print("Energies of endmembers required.")

        self._comp, partition = self.get_composition(n_atoms)
        e_end = self._energies_end @ partition
        return (energy - e_end) / np.sum(partition)

    def compute_formation_energies(
        self,
        energies: np.ndarray,
        n_atoms_array: np.ndarray,
    ):
        """Compute formation energy.

        Energy per n_atoms must be given.
        """
        if self._energies_end is None:
            print("Energies of endmembers required.")
        self._comp, partition = self.get_compositions(n_atoms_array)
        e_end = self._energies_end @ partition.T
        e_form = (energies - e_end) / np.sum(partition, axis=1)
        return e_form

    @property
    def energies_end_members(self):
        """Return energies of end members."""
        return self._energies_end

    @energies_end_members.setter
    def energies_end_members(self, e: np.ndarray):
        """Setter of energies of end members.

        Energy per chemical composition must be given.
        """
        if self._composition_axis.shape[1] != len(e):
            raise RuntimeError("Size of Energies for end members inconsitent.")
        self._energies_end = e

    @property
    def compositions(self):
        """Return compositions."""
        return self._comp

    @property
    def compositions_endmembers(self):
        """Return compositions of endmembers."""
        return np.eye(self._composition_axis.shape[1])
