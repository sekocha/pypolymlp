"""Utility class for datasets."""

from dataclasses import dataclass

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.utils import split_ids_train_test


@dataclass
class DatasetDFT:
    """Dataclass of DFT dataset used for developing polymlp.

    Parameters
    ----------
    energies: Energies, shape=(n_str).
    forces: Forces, shape=(sum(n_atom(i_str) * 3)).
    stresses: Stress tensor elements, shape=(n_str * 6).
    volumes: Volumes, shape=(n_str).
    structures: Structures, list[PolymlpStructure]
    total_n_atoms: Numbers of atoms in structures.
    files: File names of structures.
    """

    energies: np.ndarray
    forces: np.ndarray
    stresses: np.ndarray
    volumes: np.ndarray
    structures: list[PolymlpStructure]
    total_n_atoms: np.ndarray
    files: list[str]
    elements: list[str]
    include_force: bool = True
    weight: float = 1.0
    name: str = "dataset"
    exist_force: bool = True
    exist_stress: bool = True

    def __post_init__(self):
        """Post init method."""
        self.check_errors()

    def check_errors(self):
        """Check errors."""
        assert self.energies.shape[0] * 6 == self.stresses.shape[0]
        assert self.energies.shape[0] == self.volumes.shape[0]
        assert self.energies.shape[0] == len(self.structures)
        assert self.energies.shape[0] == self.total_n_atoms.shape[0]
        assert self.energies.shape[0] == len(self.files)
        assert self.forces.shape[0] == np.sum(self.total_n_atoms) * 3

    def apply_atomic_energy(self, atom_e: tuple[float]):
        """Subtract atomic energies from energies."""
        atom_e = np.array(atom_e)
        self.energies = np.array(
            [e - st.n_atoms @ atom_e for e, st in zip(self.energies, self.structures)]
        )
        return self

    def slice(self, begin: int, end: int, name: str = "sliced"):
        """Slice DFT data in DatasetDFT."""
        begin_f = sum(self.total_n_atoms[:begin]) * 3
        end_f = sum(self.total_n_atoms[:end]) * 3
        dft_dict_sliced = DatasetDFT(
            energies=self.energies[begin:end],
            forces=self.forces[begin_f:end_f],
            stresses=self.stresses[begin * 6 : end * 6],
            volumes=self.volumes[begin:end],
            structures=self.structures[begin:end],
            total_n_atoms=self.total_n_atoms[begin:end],
            files=self.files[begin:end],
            elements=self.elements,
            include_force=self.include_force,
            weight=self.weight,
            name=name,
        )
        return dft_dict_sliced

    def _force_stress_ids(self, ids: np.ndarray):
        """Return IDs for force and stress corresponding to IDs for energy."""
        force_end = np.cumsum(self.total_n_atoms * 3)
        force_begin = np.insert(force_end[:-1], 0, 0)
        ids_force = np.array(
            [i for b, e in zip(force_begin[ids], force_end[ids]) for i in range(b, e)]
        )
        ids_stress = ((ids * 6)[:, None] + np.arange(6)[None, :]).reshape(-1)
        return ids_force, ids_stress

    def sort(self):
        """Sort DFT data in terms of the number of atoms."""
        ids = np.argsort(self.total_n_atoms)
        ids_force, ids_stress = self._force_stress_ids(ids)

        self.energies = self.energies[ids]
        self.forces = self.forces[ids_force]
        self.stresses = self.stresses[ids_stress]
        self.volumes = self.volumes[ids]
        self.total_n_atoms = self.total_n_atoms[ids]
        self.structures = [self.structures[i] for i in ids]
        self.files = [self.files[i] for i in ids]
        return self

    def split(self, train_ratio: float = 0.9):
        """Split dataset into training and test datasets."""
        train_ids, test_ids = split_ids_train_test(len(self.energies))
        ids_force, ids_stress = self._force_stress_ids(train_ids)
        train = DatasetDFT(
            energies=self.energies[train_ids],
            forces=self.forces[ids_force],
            stresses=self.stresses[ids_stress],
            volumes=self.volumes[train_ids],
            structures=[self.structures[i] for i in train_ids],
            total_n_atoms=self.total_n_atoms[train_ids],
            files=[self.files[i] for i in train_ids],
            elements=self.elements,
            include_force=self.include_force,
            weight=self.weight,
            name=self.name,
        )
        ids_force, ids_stress = self._force_stress_ids(test_ids)
        test = DatasetDFT(
            energies=self.energies[test_ids],
            forces=self.forces[ids_force],
            stresses=self.stresses[ids_stress],
            volumes=self.volumes[test_ids],
            structures=[self.structures[i] for i in test_ids],
            total_n_atoms=self.total_n_atoms[test_ids],
            files=[self.files[i] for i in test_ids],
            elements=self.elements,
            include_force=self.include_force,
            weight=self.weight,
            name=self.name,
        )
        return train, test
