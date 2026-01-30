"""Class for datasets."""

import glob
from typing import List, Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.dataset_utils import DatasetDFT
from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.core.interface_yaml import (
    parse_electron_yamls,
    set_dataset_from_electron_yamls,
    set_dataset_from_sscha_yamls,
)
from pypolymlp.core.utils import split_train_test, strtobool


class Dataset:
    """Class for keeping a dataset."""

    def __init__(
        self,
        dataset_type: Literal["vasp", "sscha", "electron", "phono3py"] = "vasp",
        files: Optional[Union[list, str]] = None,
        location: Optional[str] = None,
        include_force: bool = True,
        include_stress: bool = True,
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
        include_force: Consider stress or not for the dataset.
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

        self._set_dataset_attrs(
            files=files,
            location=location,
            include_force=include_force,
            include_stress=include_stress,
            weight=weight,
            strings=strings,
            name=name,
            prefix_location=prefix_location,
        )

        self._element_order = None
        self._finished_atomic_energy = False

    def _set_dataset_attrs(
        self,
        files: Optional[Union[str, list]] = None,
        location: Optional[str] = None,
        include_force: bool = True,
        include_stress: bool = True,
        weight: float = 1.0,
        strings: Optional[list] = None,
        name: Optional[str] = None,
        prefix_location: Optional[str] = None,
    ):
        """Set attributes for dataset."""
        if files is not None:
            self._name = "Data_from_files" if name is None else name
            self._files = files
            if prefix_location is not None and isinstance(files, str):
                self._files = prefix_location + "/" + self._files
            self._include_force = include_force
            self._include_stress = include_stress
            self._weight = weight
        elif location is not None:
            self._name = location if name is None else name
            if prefix_location is not None:
                location = prefix_location + "/" + location

            self._files = sorted(glob.glob(location))
            self._include_force = include_force
            self._include_stress = include_stress
            self._weight = weight
        elif strings is not None:
            self._name = location = strings[0]
            if prefix_location is not None:
                location = prefix_location + "/" + location

            self._files = sorted(glob.glob(location))
            if len(strings) > 1:
                self._include_force = strtobool(strings[1])
            else:
                self._include_force = include_force

            if not self._include_force:
                self._include_stress = False
            else:
                self._include_stress = include_stress

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
            include_stress=self._include_stress,
            weight=self._weight,
            name="Train_" + self._name,
            split=True,
        )
        test = Dataset(
            dataset_type=self._dataset_type,
            files=files_test,
            include_force=self._include_force,
            include_stress=self._include_stress,
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
            include_stress=self._include_stress,
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
            include_stress=self._include_stress,
            weight=self._weight,
            name=test_name,
            split=True,
            dft=test_dft,
        )
        return train, test

    def slice_dft(self, begin: int, end: int):
        """Slice DFT data in dataset."""
        dft = self._dft.slice(begin, end)
        name = "Sliced_" + self._name
        data = Dataset(
            dataset_type=self._dataset_type,
            files=name,
            include_force=self._include_force,
            include_stress=self._include_stress,
            weight=self._weight,
            name=name,
            dft=dft,
        )
        return data

    def sort_dft(self):
        """Sort DFT data in terms of the number of atoms."""
        self._dft.sort()
        return self

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

        self.subtract_atomic_energy(params.atomic_energy)
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

        if self._finished_atomic_energy:
            if self._verbose:
                print(
                    "Atomic energies have already been subtracted,",
                    "given atomic energies are not applied to dataset",
                    flush=True,
                )
        else:
            self._dft.apply_atomic_energy(atomic_energy)
            self._finished_atomic_energy = True
        return self

    @property
    def dft(self):
        """Return DFT data."""
        return self._dft

    @dft.setter
    def dft(self, dft: DatasetDFT):
        """Setter of DFT data."""
        self._dft = dft

    @property
    def include_force(self):
        """Return include_force."""
        return self._include_force

    @include_force.setter
    def include_force(self, force: bool):
        """Setter of include_force."""
        self._include_force = force

    @property
    def include_stress(self):
        """Return include_stress."""
        return self._include_stress

    @include_stress.setter
    def include_stress(self, stress: bool):
        """Setter of include_stress."""
        self._include_stress = stress

    @property
    def weight(self):
        """Return weight."""
        return self._weight

    @weight.setter
    def weight(self, w: float):
        """Setter of weight."""
        self._weight = w

    @property
    def name(self):
        """Return dataset name."""
        return self._name

    @name.setter
    def name(self, n: str):
        """Setter of dataset name."""
        self._name = n

    @property
    def files(self):
        """Return file names."""
        return self._files

    @files.setter
    def files(self, f: list):
        """Setter of file names."""
        self._files = f

    @property
    def energies(self) -> np.ndarray:
        """Get energies."""
        if self._dft is None:
            return None
        return self._dft.energies

    @property
    def forces(self) -> np.ndarray:
        """Get forces."""
        if self._dft is None:
            return None
        return self._dft.forces

    @property
    def stresses(self) -> np.ndarray:
        """Get stresses."""
        if self._dft is None:
            return None
        return self._dft.stresses

    @property
    def volumes(self) -> np.ndarray:
        """Get volumes."""
        if self._dft is None:
            return None
        return self._dft.volumes

    @property
    def structures(self) -> List[PolymlpStructure]:
        """Get structures."""
        if self._dft is None:
            return None
        return self._dft.structures

    @property
    def total_n_atoms(self) -> np.ndarray:
        """Get total number of atoms."""
        if self._dft is None:
            return None
        return self._dft.total_n_atoms

    @property
    def exist_force(self) -> bool:
        """Get force existence flag."""
        if self._dft is None:
            return None
        return self._dft.exist_force

    @property
    def exist_stress(self) -> bool:
        """Get stress existence flag."""
        if self._dft is None:
            return None
        return self._dft.exist_stress


class DatasetList:
    """Class for keeping multiple datasets for training or test data."""

    def __init__(self, datasets: Optional[Union[list[Dataset], Dataset]] = None):
        """Init method."""
        if datasets is None:
            self._datasets = []
        elif isinstance(datasets, Dataset):
            self._datasets = [datasets]
        else:
            self._datasets = datasets

    def __iter__(self):
        """Iter method."""
        return iter(self._datasets)

    def __getitem__(self, index: int):
        """Getitem method."""
        return self._datasets[index]

    def __setitem__(self, index: int, value: Dataset):
        """Setitem method."""
        self._datasets[index] = value

    def __len__(self):
        """Len method."""
        return len(self._datasets)

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

    def subtract_atomic_energy(self, atomic_energy: tuple):
        """Subtract atomic energy."""
        for ds in self._datasets:
            ds.subtract_atomic_energy(atomic_energy)
        return self

    @property
    def datasets(self):
        """Return datasets."""
        return self._datasets

    @property
    def include_force(self):
        """Return include_force."""
        return np.any([ds.include_force for ds in self._datasets])


def set_datasets_from_multiple_filesets(
    params: PolymlpParams,
    train_files: Optional[list[list[str]]] = None,
    test_files: Optional[list[list[str]]] = None,
    weight: float = 1.0,
):
    """Set datasets from files and params."""
    train, test = [], []
    for i, files in enumerate(train_files):
        train.append(
            Dataset(
                name="data" + str(i + 1),
                dataset_type=params.dataset_type,
                files=sorted(files),
                include_force=params.include_force,
                include_stress=params.include_stress,
                weight=weight,
            )
        )
    for i, files in enumerate(test_files):
        test.append(
            Dataset(
                name="data" + str(i + 1),
                dataset_type=params.dataset_type,
                files=sorted(files),
                include_force=params.include_force,
                include_stress=params.include_stress,
                weight=weight,
            )
        )
    train = DatasetList(train)
    test = DatasetList(test)
    train.parse_files(params)
    test.parse_files(params)
    return train, test


def set_datasets_from_single_fileset(
    params: PolymlpParams,
    files: Optional[Union[list[str], str]] = None,
    train_files: Optional[list[str]] = None,
    test_files: Optional[list[str]] = None,
    train_ratio: float = 0.9,
    weight: float = 1.0,
):
    """Set datasets from files and params."""
    parse_end = True
    if files is None:
        train = Dataset(
            name="data1",
            dataset_type=params.dataset_type,
            files=sorted(train_files),
            include_force=params.include_force,
            include_stress=params.include_stress,
            weight=weight,
        )
        test = Dataset(
            name="data2",
            dataset_type=params.dataset_type,
            files=sorted(test_files),
            include_force=params.include_force,
            include_stress=params.include_stress,
            weight=weight,
        )
    else:
        if isinstance(files, str):
            data = Dataset(
                name="data",
                dataset_type=params.dataset_type,
                files=files,
                include_force=params.include_force,
                include_stress=params.include_stress,
                weight=weight,
            )
            data.parse_files(params)
            train, test = data.split_dft(train_ratio=train_ratio)
            train.name, test.name = "data1", "data2"
            parse_end = False
        else:
            data = Dataset(
                name="data",
                dataset_type=params.dataset_type,
                files=sorted(files),
                include_force=params.include_force,
                include_stress=params.include_stress,
                weight=weight,
            )
            train, test = data.split_files(train_ratio=train_ratio)
            train.name, test.name = "data1", "data2"

    train = DatasetList(train)
    test = DatasetList(test)
    if parse_end:
        train.parse_files(params)
        test.parse_files(params)

    train.subtract_atomic_energy(params.atomic_energy)
    test.subtract_atomic_energy(params.atomic_energy)
    return train, test


def set_datasets_from_structures(
    params: PolymlpParams,
    structures: Optional[list[PolymlpStructure]] = None,
    energies: Optional[np.ndarray] = None,
    forces: Optional[list[np.ndarray]] = None,
    stresses: Optional[np.ndarray] = None,
    train_structures: Optional[list[PolymlpStructure]] = None,
    test_structures: Optional[list[PolymlpStructure]] = None,
    train_energies: Optional[np.ndarray] = None,
    test_energies: Optional[np.ndarray] = None,
    train_forces: Optional[list[np.ndarray]] = None,
    test_forces: Optional[list[np.ndarray]] = None,
    train_stresses: Optional[np.ndarray] = None,
    test_stresses: Optional[np.ndarray] = None,
    train_ratio: float = 0.9,
    weight: float = 1.0,
):
    """Set datasets from files and params."""
    if structures is not None and energies is not None:
        dft = DatasetDFT(
            structures,
            energies,
            forces=forces,
            stresses=stresses,
            element_order=params.element_order,
        )
        data = Dataset(
            name="data",
            dataset_type=params.dataset_type,
            files="data",
            include_force=params.include_force,
            include_stress=params.include_stress,
            weight=weight,
            dft=dft,
        )
        train, test = data.split_dft(train_ratio=train_ratio)
        train.name, test.name = "data1", "data2"
    else:
        train_dft = DatasetDFT(
            train_structures,
            train_energies,
            forces=train_forces,
            stresses=train_stresses,
            element_order=params.element_order,
        )
        test_dft = DatasetDFT(
            test_structures,
            test_energies,
            forces=test_forces,
            stresses=test_stresses,
            element_order=params.element_order,
        )
        train = Dataset(
            name="data1",
            dataset_type=params.dataset_type,
            files="data1",
            include_force=params.include_force,
            include_stress=params.include_stress,
            weight=weight,
            dft=train_dft,
        )
        test = Dataset(
            name="data2",
            dataset_type=params.dataset_type,
            files="data2",
            include_force=params.include_force,
            include_stress=params.include_stress,
            weight=weight,
            dft=test_dft,
        )
    train = DatasetList(train)
    test = DatasetList(test)

    train.subtract_atomic_energy(params.atomic_energy)
    test.subtract_atomic_energy(params.atomic_energy)
    return train, test
