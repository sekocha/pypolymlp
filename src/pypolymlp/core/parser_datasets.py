"""Class of parsing DFT datasets."""

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.dataset import Dataset
from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.core.interface_yaml import (
    set_dataset_from_electron_yamls,
    set_dataset_from_sscha_yamls,
)


class ParserDatasets:
    """Class of parsing DFT datasets."""

    def __init__(self, params: PolymlpParams, train_ratio: float = 0.9):
        """Init method."""
        self._params = params
        self._dataset_type = self._params.dataset_type
        self._train_ratio = train_ratio

        self._train = None
        self._test = None
        self._parse()

    def _parse(self):
        """Parse datasets."""
        self._train, self._test = [], []
        if self._dataset_type == "vasp":
            self._parse_vasp()
        elif self._dataset_type == "phono3py":
            self._parse_phono3py()
        elif self._dataset_type == "sscha":
            self._parse_sscha()
        elif self._dataset_type == "electron":
            self._parse_electron()
        else:
            raise KeyError("Given dataset_type is unavailable.")

    def _inherit_dataset_params(self, dataset: Dataset, dft: PolymlpDataDFT):
        """Inherit parameters of dataset."""
        dft.apply_atomic_energy(self._params.atomic_energy)
        dft.name = dataset.name
        dft.include_force = dataset.include_force
        dft.weight = dataset.weight
        return dft

    def _parse_vasp(self):
        """Parse VASP multiple datasets."""
        for dataset in self._params.dft_train:
            dft = set_dataset_from_vaspruns(
                dataset.files,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)
            if dataset.split:
                self._train.append(dft)
            else:
                train, test = dft.split(train_ratio=self._train_ratio)
                self._train.append(train)
                self._test.append(test)

        for dataset in self._params.dft_test:
            dft = set_dataset_from_vaspruns(
                dataset.files,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)
            if dataset.split:
                self._test.append(dft)

    def _parse_phono3py(self):
        """Parse phono3py dataset."""
        from pypolymlp.core.interface_phono3py import parse_phono3py_yaml

        for dataset in self._params.dft_train:
            dft = parse_phono3py_yaml(
                dataset.location,
                energies_filename=dataset.energy_dat,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)

            train, test = dft.split(train_ratio=self._train_ratio)
            self._train.append(train)
            self._test.append(test)

    def _parse_sscha(self):
        """Parse sscha results."""
        for dataset in self._params.dft_train:
            dft = set_dataset_from_sscha_yamls(
                dataset.files,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)
            self._train.append(dft)

        for dataset in self._params.dft_test:
            dft = set_dataset_from_sscha_yamls(
                dataset.files,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)
            self._test.append(dft)

    def _parse_electron(self):
        """Parse electron results."""
        for dataset in self._params.dft_train:
            dft = set_dataset_from_electron_yamls(
                dataset.files,
                temperature=self._params.temperature,
                target=self._params.electron_property,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)
            self._train.append(dft)

        for dataset in self._params.dft_test:
            dft = set_dataset_from_electron_yamls(
                dataset.files,
                temperature=self._params.temperature,
                target=self._params.electron_property,
                element_order=self._params.element_order,
            )
            dft = self._inherit_dataset_params(dataset, dft)
            self._test.append(dft)

    @property
    def train(self) -> list[PolymlpDataDFT]:
        """Return DFT datasets for training."""
        return self._train

    @property
    def test(self) -> list[PolymlpDataDFT]:
        """Return DFT datasets for test."""
        return self._test

    @property
    def is_multiple_datasets(self) -> bool:
        """Return whether multiple datasets are considered."""
        return True

    @property
    def dataset_type(self) -> str:
        """Return dataset type."""
        return self._dataset_type
