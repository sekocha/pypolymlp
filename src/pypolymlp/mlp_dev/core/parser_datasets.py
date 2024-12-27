"""Class of parsing DFT datasets."""

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpParams
from pypolymlp.core.interface_vasp import set_dataset_from_vaspruns
from pypolymlp.core.interface_yaml import (
    set_dataset_from_electron_yamls,
    set_dataset_from_sscha_yamls,
)


class ParserDatasets:
    """Class of parsing DFT datasets."""

    def __init__(self, params: PolymlpParams):
        """Init method."""
        self._params = params
        self._dataset_type = self._params.dataset_type

        self._train = None
        self._test = None
        self._parse()

    def _parse(self):
        """Parse datasets."""
        if self._dataset_type == "vasp":
            self._parse_vasp_multiple()
        elif self._dataset_type == "phono3py":
            self._parse_phono3py_single()
        elif self._dataset_type == "sscha":
            self._parse_sscha_single()
        elif self._dataset_type == "electron":
            self._parse_electron_single()
        else:
            raise KeyError("Given dataset_type is unavailable.")

    def _post_single_dataset(self):
        self._train.name = "train_single"
        self._train.include_force = self._params.include_force
        self._train.apply_atomic_energy(self._params.atomic_energy)

        self._test.name = "test_single"
        self._test.include_force = self._params.include_force
        self._test.apply_atomic_energy(self._params.atomic_energy)

        self._train = [self._train]
        self._test = [self._test]

    def _parse_vasp_single(self):
        """Parse VASP single dataset."""
        self._train = set_dataset_from_vaspruns(
            self._params.dft_train,
            element_order=self._params.element_order,
        )
        self._test = set_dataset_from_vaspruns(
            self._params.dft_test,
            element_order=self._params.element_order,
        )
        self._post_single_dataset()

    def _parse_vasp_multiple(self):
        """Parse VASP multiple datasets."""
        element_order = self._params.element_order
        self._train = []
        for name, dict1 in self._params.dft_train.items():
            dft = set_dataset_from_vaspruns(
                dict1["vaspruns"],
                element_order=element_order,
            )
            dft.apply_atomic_energy(self._params.atomic_energy)
            dft.name = name
            dft.include_force = dict1["include_force"]
            dft.weight = dict1["weight"]
            self._train.append(dft)

        self._test = []
        for name, dict1 in self._params.dft_test.items():
            dft = set_dataset_from_vaspruns(
                dict1["vaspruns"],
                element_order=element_order,
            )
            dft.apply_atomic_energy(self._params.atomic_energy)
            dft.name = name
            dft.include_force = dict1["include_force"]
            dft.weight = dict1["weight"]
            self._test.append(dft)

    def _parse_phono3py_single(self):
        """Parse phono3py single dataset."""
        from pypolymlp.core.interface_phono3py import parse_phono3py_yaml

        self._train = parse_phono3py_yaml(
            self._params.dft_train["phono3py_yaml"],
            self._params.dft_train["energy"],
            element_order=self._params.element_order,
            select_ids=self._params.dft_train["indices"],
            use_phonon_dataset=False,
        )
        self._test = parse_phono3py_yaml(
            self._params.dft_test["phono3py_yaml"],
            self._params.dft_test["energy"],
            element_order=self._params.element_order,
            select_ids=self._params.dft_test["indices"],
            use_phonon_dataset=False,
        )
        self._post_single_dataset()
        self._params.dft_train = {"train_phono3py": self._params.dft_train}
        self._params.dft_test = {"test_phono3py": self._params.dft_test}

    def _parse_phono3py_multiple(self):
        raise NotImplementedError("No function for parsing multiple phono3py.yamls.")

    def _parse_sscha_single(self):
        """Parse sscha results."""
        self._train = set_dataset_from_sscha_yamls(
            self._params.dft_train,
            element_order=self._params.element_order,
        )
        self._test = set_dataset_from_sscha_yamls(
            self._params.dft_test,
            element_order=self._params.element_order,
        )
        self._post_single_dataset()
        self._params.dft_train = {"train_single": self._params.dft_train}
        self._params.dft_test = {"test_single": self._params.dft_test}

    def _parse_electron_single(self):
        """Parse electron results."""
        self._train = set_dataset_from_electron_yamls(
            self._params.dft_train,
            temperature=self._params.temperature,
            element_order=self._params.element_order,
        )
        self._test = set_dataset_from_electron_yamls(
            self._params.dft_test,
            temperature=self._params.temperature,
            element_order=self._params.element_order,
        )
        self._post_single_dataset()
        self._params.dft_train = {"train_single": self._params.dft_train}
        self._params.dft_test = {"test_single": self._params.dft_test}

    @property
    def train(self) -> list[PolymlpDataDFT]:
        return self._train

    @property
    def test(self) -> list[PolymlpDataDFT]:
        return self._test

    @property
    def is_multiple_datasets(self) -> bool:
        return True

    @property
    def dataset_type(self) -> str:
        return self._dataset_type
