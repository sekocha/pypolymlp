"""Class for datasets."""

import glob
from typing import Literal, Optional, Union

from pypolymlp.core.data_format import PolymlpParams
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

            self._files = sorted(glob.glob(location))
            self._include_force = include_force
            self._weight = weight
        elif strings is not None:
            self._name = location = strings[0]
            if prefix_location is not None:
                location = prefix_location + location

            self._files = sorted(glob.glob(location))
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

        self._dft.apply_atomic_energy(atomic_energy)
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


def set_datasets(files: list[str], params: PolymlpParams, train_ratio: float = 0.9):
    """Set datasets from files and params."""
    data = Dataset(
        name="data",
        dataset_type=params.dataset_type,
        files=files,
        include_force=params.include_force,
        weight=1.0,
    )
    train, test = data.split_files(train_ratio=train_ratio)
    train.name, test.name = "data1", "data2"
    train = DatasetList(train)
    test = DatasetList(test)
    train.parse_files(params)
    test.parse_files(params)
    return train, test
