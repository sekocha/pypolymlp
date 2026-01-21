"""Class for datasets."""

import glob
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.core.interface_yaml import (
    parse_electron_yamls,
    set_dataset_from_electron_yamls,
    set_dataset_from_sscha_yamls,
)
from pypolymlp.core.utils import split_ids_train_test, split_train_test, strtobool


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


class Dataset:
    """Class for keeping a dataset."""

    def __init__(
        self,
        dataset_type: Literal["vasp", "sscha", "electron", "phono3py"] = "vasp",
        files: Optional[Union[list, str]] = None,
        location: Optional[str] = None,
        include_force: bool = True,
        weight: float = 1.0,
        strings: Optional[list] = None,
        name: Optional[str] = None,
        split: bool = True,
        prefix_location: Optional[str] = None,
        dft: Optional[DatasetDFT] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        dataset_type: Dataset type.
        strings: String list for specifying dataset file locations,
                     include_force applied to the dataset,
                     and weight applied to the dataset.
        location: String for specifying dataset files.
        files: Dataset files.
        include_force: Consider force or not for the dataset.
        weight: Weight for the dataset.
        name: Name of dataset.
        split: Dataset is obtained by splitting into training and test datasets.
        """
        if strings is None and location is None and files is None:
            raise RuntimeError("Strings, location and files not found.")

        self._dataset_type = dataset_type
        self._split = split
        self._dft = dft
        self._verbose = verbose
        self._use_single_file = False

        if dataset_type == "phono3py":
            self._use_single_file = True
            if strings is not None:
                files = strings[0]
            if files is None:
                raise RuntimeError("Files are required for dataset_type phono3py.")
            if not isinstance(files, str):
                raise RuntimeError("Files must be string for dataset_type phono3py.")

        if self._dft is None:
            self._set_dataset_attrs(
                files=files,
                location=location,
                include_force=include_force,
                weight=weight,
                strings=strings,
                name=name,
                prefix_location=prefix_location,
            )

        self._element_order = None

    def _set_dataset_attrs(
        self,
        files: Optional[list] = None,
        location: Optional[str] = None,
        include_force: bool = True,
        weight: float = 1.0,
        strings: Optional[list] = None,
        name: Optional[str] = None,
        prefix_location: Optional[str] = None,
    ):
        """Set attributes for dataset."""
        if files is not None:
            self._name = "Data_from_files" if name is None else name
            self._files = files
            self._include_force = include_force
            self._weight = weight
        elif location is not None:
            self._name = location if name is None else name
            if prefix_location is not None:
                location = prefix_location + location

            files = sorted(glob.glob(location))
            self._include_force = include_force
            self._weight = weight
        elif strings is not None:
            self._name = location = strings[0]
            if prefix_location is not None:
                location = prefix_location + location

            files = sorted(glob.glob(location))
            if len(strings) > 1:
                self._include_force = strtobool(strings[1])
            else:
                self._include_force = include_force
            if len(strings) > 2:
                self._weight = float(strings[2])
            else:
                self._weight = weight
        return self

    def split_files(self, train_ratio: float = 0.9):
        """Split files in dataset into training and test datasets."""
        if self._dft is not None:
            raise RuntimeError("This function must be used before setting properties.")

        if self._use_single_file:
            raise RuntimeError("Data that can be split not found.")

        files_train, files_test = split_train_test(
            self._files,
            train_ratio=train_ratio,
        )
        train = Dataset(
            dataset_type=self._dataset_type,
            files=files_train,
            include_force=self._include_force,
            weight=self._weight,
            name="Train_" + self._name,
            split=True,
        )
        test = Dataset(
            dataset_type=self._dataset_type,
            files=files_test,
            include_force=self._include_force,
            weight=self._weight,
            name="Test_" + self._name,
            split=True,
        )
        return train, test

    def split_dft(self, train_ratio: float = 0.9):
        """Split DFT data in dataset into training and test datasets."""
        if self._dft is None:
            raise RuntimeError("DFT data not found.")

        train_dft, test_dft = self._dft.split(train_ratio=train_ratio)
        train_name = "Train_" + self._name
        train = Dataset(
            dataset_type=self._dataset_type,
            files=train_name,
            include_force=self._include_force,
            weight=self._weight,
            name=train_name,
            split=True,
            dft=train_dft,
        )
        test_name = "Test_" + self._name
        test = Dataset(
            dataset_type=self._dataset_type,
            files=test_name,
            include_force=self._include_force,
            weight=self._weight,
            name=test_name,
            split=True,
            dft=test_dft,
        )
        return train, test

    def parse_files(self, params: PolymlpParams):
        """Parse data from files."""
        self._element_order = params.element_order
        if self._dataset_type == "vasp":
            self._parse_vasp()
        elif self._dataset_type == "phono3py":
            self._parse_phono3py()
        elif self._dataset_type == "sscha":
            self._parse_sscha()
        elif self._dataset_type == "electron":
            self._parse_electron(params)
        else:
            raise KeyError("Given dataset_type is unavailable.")

        self.subtract_atomic_energy(params)
        return self

    def _parse_vasp(self):
        """Parse data from vaspruns."""
        self._dft = set_dataset_from_vaspruns(
            self._files,
            element_order=self._element_order,
        )
        return self

    def _parse_phono3py(self):
        """Parse data from phono3py.yaml."""
        from pypolymlp.core.interface_phono3py import parse_phono3py_yaml

        self._dft = parse_phono3py_yaml(
            self._files,
            element_order=self._element_order,
        )
        return self

    def _parse_sscha(self):
        """Parse data from polymlp_sscha.yaml files."""
        self._dft = set_dataset_from_sscha_yamls(
            self._files,
            element_order=self._element_order,
        )
        return self

    def _parse_electron(self, params: PolymlpParams):
        """Parse data from electron.yaml."""
        # TODO: Efficient implementation for multiple temperatures and properties.
        yml_data = parse_electron_yamls(self._files)
        self._dft = set_dataset_from_electron_yamls(
            yml_data,
            temperature=params.temperature,
            target=params.electron_property,
            element_order=self._element_order,
        )
        return self

    def subtract_atomic_energy(self, atomic_energy: tuple):
        """Subtract atomic energy."""
        if self._dft is None:
            raise RuntimeError("DFT data not found.")

        self._dft.apply_atomic_energy(atomic_energy)
        return self

    @property
    def dft(self):
        """Return DFT data."""
        return self._dft

    @property
    def include_force(self):
        """Return include_force."""
        return self._include_force

    @include_force.setter
    def include_force(self, force: bool):
        """Setter of include_force."""
        self._include_force = force

    @property
    def weight(self):
        """Return weight."""
        return self._weight

    @property
    def name(self):
        """Return dataset name."""
        return self._name


class DatasetList:
    """Class for keeping multiple datasets for training or test data."""

    def __init__(self, datasets: Optional[list[Dataset]] = None):
        """Init method."""
        self._datasets = [] if datasets is None else datasets

    def __iter__(self):
        """Iter method."""
        return iter(self._datasets)

    def __getitem__(self, index: int):
        """Getitem method."""
        return self._datasets[index]

    def __setitem__(self, index: int, value: Dataset):
        """Setitem method."""
        self._datasets[index] = value

    def append(self, datasets: Union[list, tuple, Dataset]):
        """Append dataset."""
        if isinstance(datasets, Dataset):
            self._datasets.append(datasets)
        elif isinstance(datasets, DatasetList):
            self._datasets.extend(datasets._datasets)
        elif isinstance(datasets, (list, tuple)):
            self._datasets.extend(datasets)
        return self

    def parse_files(self, params: PolymlpParams):
        """Parse files in datasets."""
        for ds in self._datasets:
            ds.parse_files(params)
        return self

    @property
    def datasets(self):
        """Return datasets."""
        return self._datasets
