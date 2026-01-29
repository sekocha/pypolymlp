"""Class of DFT dataset used for developing polymlp."""

import copy
from typing import List, Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.utils import split_ids_train_test


def permute_atoms(
    st: PolymlpStructure,
    force: np.ndarray,
    element_order: list[str],
) -> tuple[PolymlpStructure, np.ndarray]:
    """Permute atoms in structure and forces.

    The orders of atoms and forces are compatible with the element order.
    """
    positions, n_atoms, elements, types = [], [], [], []
    force_permute = []
    for atomtype, ele in enumerate(element_order):
        ids = np.where(np.array(st.elements) == ele)[0]
        n_match = len(ids)
        positions.extend(st.positions[:, ids].T)
        n_atoms.append(n_match)
        elements.extend([ele for _ in range(n_match)])
        types.extend([atomtype for _ in range(n_match)])
        force_permute.extend(force[:, ids].T)

    positions = np.array(positions).T
    force_permute = np.array(force_permute).T

    st_new = copy.deepcopy(st)
    st_new.positions = positions
    st_new.n_atoms = n_atoms
    st_new.elements = elements
    st_new.types = types
    return st_new, force_permute


class DatasetDFT:
    """Class of DFT dataset used for developing polymlp."""

    def __init__(
        self,
        structures: Optional[list[PolymlpStructure]] = None,
        energies: Optional[np.ndarray] = None,
        forces: Optional[List[np.ndarray]] = None,
        stresses: Optional[np.ndarray] = None,
        element_order: Optional[List[str]] = None,
    ):
        """Init method.

        Parameters
        ----------
        structures: Structures, list[PolymlpStructure].
        energies: Energies, shape=(n_str).
        forces: Forces, shape=(n_str, 3, n_atom(i_str)).
        stresses: Stress tensor elements, shape=(n_str, 3, 3).
        element_order: Order of elements to define atom types.

        Attributes
        ----------
        structures: Structures, list[PolymlpStructure]
        energies: Energies, shape=(n_str).
        forces: Forces, shape=(sum(n_atom(i_str) * 3)).
        stresses: Stress tensor elements, shape=(n_str * 6).
        volumes: Volumes, shape=(n_str).
        total_n_atoms: Numbers of atoms in structures.
        elements: List of element symbols.
        files: List of structure names or file names.
        exist_force: Whether force data exists.
        exist_stress: Whether stress data exists.
        """
        self._exist_force = None
        self._exist_stress = None
        if structures is not None and energies is not None:
            self._set_properties(
                structures=structures,
                energies=energies,
                forces=forces,
                stresses=stresses,
                element_order=element_order,
            )

    def _set_properties(
        self,
        structures: list[PolymlpStructure],
        energies: np.ndarray,
        forces: Optional[List[np.ndarray]] = None,
        stresses: Optional[np.ndarray] = None,
        element_order: Optional[List[str]] = None,
    ):
        """Set properties in the form used for pypolymlp regression."""
        assert len(structures) == len(energies)
        if forces is not None:
            assert len(structures) == len(forces)
        if stresses is not None:
            assert len(structures) == len(stresses)

        self._exist_force = True if forces is not None else False
        self._exist_stress = True if stresses is not None else False
        if forces is None:
            forces = [np.zeros((3, len(st.elements))) for st in structures]
        if stresses is None:
            stresses = [np.zeros((3, 3)) for _ in energies]

        structures_data, forces_data, stresses_data = [], [], []
        for st, force_st, sigma in zip(structures, forces, stresses):
            if element_order is not None:
                st1, force_st1 = permute_atoms(st, force_st, element_order)
            else:
                st1, force_st1 = st, force_st

            structures_data.append(st1)
            force_ravel = np.ravel(force_st1, order="F")
            forces_data.extend(force_ravel)
            stresses_data.extend(
                [
                    sigma[0][0],
                    sigma[1][1],
                    sigma[2][2],
                    sigma[0][1],
                    sigma[1][2],
                    sigma[2][0],
                ]
            )

        self._structures = structures_data
        self._energies = np.array(energies)
        self._forces = np.array(forces_data)
        self._stresses = np.array(stresses_data)
        self._volumes = np.array([st.volume for st in self._structures])
        self._total_n_atoms = np.array([sum(st.n_atoms) for st in self._structures])
        self._files = [st.name for st in self._structures]

        if element_order is None:
            # This part must be tested. In general, element_order is not None.
            elements_size = [len(st.n_atoms) for st in self._structures]
            elements = self._structures[np.argmax(elements_size)].elements
            self._elements = sorted(set(elements), key=elements.index)
        else:
            self._elements = element_order

        self._check_errors()
        return self

    def _check_errors(self):
        """Check errors."""
        assert self._energies.shape[0] * 6 == self._stresses.shape[0]
        assert self._energies.shape[0] == self._volumes.shape[0]
        assert self._energies.shape[0] == len(self._structures)
        assert self._energies.shape[0] == self._total_n_atoms.shape[0]
        assert self._forces.shape[0] == np.sum(self._total_n_atoms) * 3
        return self

    def assign(
        self,
        structures,
        energies,
        forces,
        stresses,
        volumes,
        total_n_atoms,
        elements,
        files,
        exist_force,
        exist_stress,
    ):
        """Assign all attributes."""
        self._structures = structures
        self._energies = energies
        self._forces = forces
        self._stresses = stresses
        self._volumes = volumes
        self._total_n_atoms = total_n_atoms
        self._elements = elements
        self._files = files
        self._exist_force = exist_force
        self._exist_stress = exist_stress
        return self

    def apply_atomic_energy(self, atom_e: tuple[float]):
        """Subtract atomic energies from energies."""
        atom_e = np.array(atom_e)
        zip_obj = zip(self._energies, self._structures)
        self._energies = np.array([e - st.n_atoms @ atom_e for e, st in zip_obj])
        return self

    def slice(self, begin: int, end: int):
        """Slice DFT data in DatasetDFT."""
        begin_f = sum(self._total_n_atoms[:begin]) * 3
        end_f = sum(self._total_n_atoms[:end]) * 3

        sliced = DatasetDFT()
        sliced.assign(
            self._structures[begin:end],
            self._energies[begin:end],
            self._forces[begin_f:end_f],
            self._stresses[begin * 6 : end * 6],
            self._volumes[begin:end],
            self._total_n_atoms[begin:end],
            self._elements,
            self._files[begin:end],
            self._exist_force,
            self._exist_stress,
        )
        return sliced

    def _force_stress_ids(self, ids: np.ndarray):
        """Return IDs for force and stress corresponding to IDs for energy."""
        force_end = np.cumsum(self._total_n_atoms * 3)
        force_begin = np.insert(force_end[:-1], 0, 0)
        ids_force = np.array(
            [i for b, e in zip(force_begin[ids], force_end[ids]) for i in range(b, e)]
        )
        ids_stress = ((ids * 6)[:, None] + np.arange(6)[None, :]).reshape(-1)
        return ids_force, ids_stress

    def sort(self):
        """Sort DFT data in terms of the number of atoms."""
        ids = np.argsort(self._total_n_atoms)
        ids_force, ids_stress = self._force_stress_ids(ids)

        self._structures = [self._structures[i] for i in ids]
        self._energies = self._energies[ids]
        self._forces = self._forces[ids_force]
        self._stresses = self._stresses[ids_stress]
        self._volumes = self._volumes[ids]
        self._total_n_atoms = self._total_n_atoms[ids]
        self._files = [self._files[i] for i in ids]
        return self

    def split(self, train_ratio: float = 0.9):
        """Split dataset into training and test datasets."""
        n_data = len(self._energies)
        train_ids, test_ids = split_ids_train_test(n_data, train_ratio=train_ratio)

        train, test = None, None
        if len(train_ids) > 0:
            ids_force, ids_stress = self._force_stress_ids(train_ids)
            train = DatasetDFT()
            train.assign(
                [self._structures[i] for i in train_ids],
                self._energies[train_ids],
                self._forces[ids_force],
                self._stresses[ids_stress],
                self._volumes[train_ids],
                self._total_n_atoms[train_ids],
                self._elements,
                [self._files[i] for i in train_ids],
                self._exist_force,
                self._exist_stress,
            )
        if len(test_ids) > 0:
            ids_force, ids_stress = self._force_stress_ids(test_ids)
            test = DatasetDFT()
            test.assign(
                [self._structures[i] for i in test_ids],
                self._energies[test_ids],
                self._forces[ids_force],
                self._stresses[ids_stress],
                self._volumes[test_ids],
                self._total_n_atoms[test_ids],
                self._elements,
                [self._files[i] for i in test_ids],
                self._exist_force,
                self._exist_stress,
            )
        return train, test

    @property
    def energies(self) -> np.ndarray:
        """Get energies."""
        return self._energies

    @energies.setter
    def energies(self, value: np.ndarray):
        """Set energies."""
        self._energies = value

    @property
    def forces(self) -> np.ndarray:
        """Get forces."""
        return self._forces

    @forces.setter
    def forces(self, value: np.ndarray):
        """Set forces."""
        self._forces = value

    @property
    def stresses(self) -> np.ndarray:
        """Get stresses."""
        return self._stresses

    @stresses.setter
    def stresses(self, value: np.ndarray):
        """Set stresses."""
        self._stresses = value

    @property
    def volumes(self) -> np.ndarray:
        """Get volumes."""
        return self._volumes

    @volumes.setter
    def volumes(self, value: np.ndarray):
        """Set volumes."""
        self._volumes = value

    @property
    def structures(self) -> List[PolymlpStructure]:
        """Get structures."""
        return self._structures

    @structures.setter
    def structures(self, value: List[PolymlpStructure]):
        """Set structures."""
        self._structures = value

    @property
    def total_n_atoms(self) -> np.ndarray:
        """Get total number of atoms."""
        return self._total_n_atoms

    @total_n_atoms.setter
    def total_n_atoms(self, value: np.ndarray):
        """Set total number of atoms."""
        self._total_n_atoms = value

    @property
    def elements(self) -> List[str]:
        """Get elements."""
        return self._elements

    @elements.setter
    def elements(self, value: List[str]):
        """Set elements."""
        self._elements = value

    @property
    def files(self) -> List[str]:
        """Get files."""
        return self._files

    @files.setter
    def files(self, value: List[str]):
        """Set files."""
        self._files = value

    @property
    def exist_force(self) -> bool:
        """Get force existence flag."""
        return self._exist_force

    @exist_force.setter
    def exist_force(self, value: bool):
        """Set force existence flag."""
        self._exist_force = value

    @property
    def exist_stress(self) -> bool:
        """Get stress existence flag."""
        return self._exist_stress

    @exist_stress.setter
    def exist_stress(self, value: bool):
        """Set stress existence flag."""
        self._exist_stress = value
