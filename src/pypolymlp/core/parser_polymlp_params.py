"""Class of input parameter parser."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParamsSingle
from pypolymlp.core.dataset import Dataset, DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.core.params_utils import (
    set_active_gaussian_params,
    set_gaussian_params,
    set_gtinv_params,
    set_regression_alphas,
)
from pypolymlp.core.parser_infile import InputParser
from pypolymlp.core.units import HartreetoEV


class ParamsParserSingle:
    """Class of input parameter parser."""

    def __init__(self, filename: str, verbose: bool = False):
        """Init method.

        Parameters
        ----------
        filename: File of input parameters for single polymlp (e.g., polymlp.in).
        """
        self._parser = InputParser(filename)
        self._verbose = verbose

        self._params = None
        self._train = None
        self._test = None

    def run(
        self,
        train_ratio: float = 0.9,
        prefix_data_location: Optional[str] = None,
    ):
        """Parse all required files."""
        self.set_params()
        self.set_datasets(
            train_ratio=train_ratio,
            prefix_data_location=prefix_data_location,
        )
        return self

    def set_params(self):
        """Get parameters from file and set them."""
        include_force, include_stress = self._get_force_tags()
        elements, n_type, atom_e, enable_spins = self._get_element_properties()
        alphas = self._get_regression_params()
        model = self._get_potential_model_params(n_type, elements)
        dataset_type = self._parser.get_params("dataset_type", default="vasp")

        self._params = PolymlpParamsSingle(
            n_type=n_type,
            elements=elements,
            model=model,
            atomic_energy=atom_e,
            enable_spins=enable_spins,
            regression_alpha=alphas,
            include_force=include_force,
            include_stress=include_stress,
            dataset_type=dataset_type,
        )

        if dataset_type == "electron":
            self._params.temperature = self._parser.get_params(
                "temperature", default=300, dtype=float
            )
            self._params.electron_property = self._parser.get_params(
                "electron_property", default="free_energy", dtype=str
            )

        return self._params

    def _get_force_tags(self):
        """Return include_force and include_stress."""
        include_force = self._parser.get_params(
            "include_force", default=True, dtype=bool
        )
        if include_force:
            include_stress = self._parser.get_params(
                "include_stress", default=True, dtype=bool
            )
        else:
            include_stress = False

        self._include_force = include_force
        return include_force, include_stress

    def _get_element_properties(self):
        """Return properties for identifying elements."""
        n_type = self._parser.get_params("n_type", default=1, dtype=int)
        elements = self._parser.get_params(
            "elements",
            size=n_type,
            default=None,
            required=True,
            dtype=str,
            return_array=True,
        )
        d_atom_e = [0.0 for i in range(n_type)]

        atom_e = self._parser.get_params(
            "atomic_energy",
            size=n_type,
            default=d_atom_e,
            dtype=float,
            return_array=True,
        )
        atom_e_unit = self._parser.get_params(
            "atomic_energy_unit",
            default="eV",
            dtype=str,
        )
        if atom_e_unit in ("Hartree", "hartree"):
            atom_e = [e * HartreetoEV for e in atom_e]

        enable_spins = self._parser.get_params(
            "enable_spins",
            size=n_type,
            default=None,
            dtype=bool,
            return_array=True,
        )
        return elements, n_type, tuple(atom_e), enable_spins

    def _get_regression_params(self):
        """Set regularization parameters in regression."""
        alpha_params = self._parser.get_params(
            "reg_alpha_params",
            size=3,
            default=(-3.0, 1.0, 5),
            dtype=float,
            return_array=True,
        )
        alphas = set_regression_alphas(alpha_params)
        return alphas

    def _get_potential_model_params(self, n_type: int, elements: tuple):
        """Set parameters for identifying potential model."""
        cutoff = self._parser.get_params("cutoff", default=6.0, dtype=float)
        model_type = self._parser.get_params("model_type", default=1, dtype=int)
        max_p = self._parser.get_params("max_p", default=1, dtype=int)
        feature_type = self._parser.get_params("feature_type", default="gtinv")

        gtinv_params, max_l = self._get_gtinv_params(n_type, feature_type)
        pair_params, pair_params_active, pair_cond = self._get_pair_params(
            cutoff, elements
        )

        model = PolymlpModelParams(
            cutoff,
            model_type,
            max_p,
            max_l,
            feature_type=feature_type,
            gtinv=gtinv_params,
            pair_type="gaussian",
            pair_conditional=pair_cond,
            pair_params=pair_params,
            pair_params_conditional=pair_params_active,
        )
        return model

    def _get_gtinv_params(self, n_type: int, feature_type: Literal["gtinv", "pair"]):
        """Set parameters for group-theoretical invariants."""
        version = self._parser.get_params("gtinv_version", default=1, dtype=int)
        order = self._parser.get_params("gtinv_order", default=3, dtype=int)
        size = order - 1
        gtinv_maxl = self._parser.get_params(
            "gtinv_maxl",
            size=size,
            default=[2 for i in range(size)],
            dtype=int,
            return_array=True,
        )
        if len(gtinv_maxl) < size:
            size_gap = size - len(gtinv_maxl)
            gtinv_maxl.extend([2 for i in range(size_gap)])

        gtinv_params, max_l = set_gtinv_params(
            n_type,
            feature_type=feature_type,
            gtinv_order=order,
            gtinv_maxl=gtinv_maxl,
            gtinv_version=version,
        )
        return gtinv_params, max_l

    def _get_pair_params(self, cutoff: float, elements: tuple):
        """Set parameters for Gaussian radial functions."""
        params1 = self._parser.get_params(
            "gaussian_params1",
            size=3,
            default=(1.0, 1.0, 1),
            dtype=float,
            return_array=True,
        )
        params2 = self._parser.get_params(
            "gaussian_params2",
            size=3,
            default=(0.0, cutoff - 1.0, 7),
            dtype=float,
            return_array=True,
        )
        n_gaussians = self._parser.get_params(
            "n_gaussians",
            default=None,
            dtype=int,
        )
        distance = self._parser.distance
        pair_params = set_gaussian_params(
            params1=params1,
            params2=params2,
            n_gaussians=n_gaussians,
            cutoff=cutoff,
        )
        pair_params_active, cond = set_active_gaussian_params(
            pair_params,
            elements,
            distance,
        )
        return pair_params, pair_params_active, cond

    def set_datasets(
        self,
        train_ratio: float = 0.9,
        prefix_data_location: Optional[str] = None,
    ):
        """Set datasets."""
        if self._params is None:
            raise RuntimeError("Use set_params at first.")

        dataset_type = self._params.dataset_type
        train_set, test_set, not_split_set = [], [], []
        for strings_all in self._parser.dataset_strings:
            tag, strings = strings_all[0], strings_all[1:]
            dataset = Dataset(
                strings=strings,
                dataset_type=dataset_type,
                prefix_location=prefix_data_location,
                verbose=self._verbose,
            )
            if not self._params.include_force:
                dataset.include_force = False
            if not self._params.include_stress:
                dataset.include_stress = False

            if tag == "train_data":
                train_set.append(dataset)
            elif tag == "test_data":
                test_set.append(dataset)
            elif tag == "data":
                try:
                    train, test = dataset.split_files(train_ratio=train_ratio)
                    train_set.append(train)
                    test_set.append(test)
                except:
                    not_split_set.append(dataset)
            elif tag == "data_md":
                not_split_set.append(dataset)

        self._train = DatasetList(train_set)
        self._test = DatasetList(test_set)
        not_split_data = DatasetList(not_split_set)

        if dataset_type == "sscha":
            self._train = self._train.split_imaginary(weight_imag=0.01)
            self._test = self._test.split_imaginary(weight_imag=0.01)
            # self._train = self._train.split_imaginary(weight_imag=1e-8)
            # self._test = self._test.split_imaginary(weight_imag=1e-8)

        self._train.parse_files(self._params)
        self._test.parse_files(self._params)
        not_split_data.parse_files(self._params)

        for dataset in not_split_data:
            train, test = dataset.split_dft(train_ratio=train_ratio)
            self._train.append(train)
            self._test.append(test)

        return self

    @property
    def params(self) -> PolymlpParamsSingle:
        """Return parameters for developing polymlp."""
        return self._params

    @property
    def train(self) -> DatasetList:
        """Return training datasets."""
        return self._train

    @property
    def test(self) -> DatasetList:
        """Return test datasets."""
        return self._test


class ParamsParser:
    """Class of input parameter parser."""

    def __init__(
        self,
        infiles: Union[str, list[str]],
        train_ratio: float = 0.9,
        prefix_data_location: Optional[str] = None,
        parse_dft: bool = True,
        verbose: bool = False,
    ):
        """Init method."""
        self._train_ratio = train_ratio
        self._prefix_data_location = prefix_data_location
        self._parse_dft = parse_dft
        self._verbose = verbose

        self._priority_file = None
        self._params = None
        self._train = None
        self._test = None

        if isinstance(infiles, str):
            self._set_from_single_file(infiles)
        elif isinstance(infiles, (list, tuple, np.ndarray)):
            self._set_from_multiple_files(infiles)
        else:
            raise RuntimeError("Inappropriate format for input files.")

    def _set_from_single_file(self, infile: str):
        """Set parameters from single input file."""
        self._priority_file = infile
        parser = ParamsParserSingle(infile, verbose=self._verbose)
        parser.set_params()
        self._params = PolymlpParams(parser.params)

        if self._parse_dft:
            parser.set_datasets(
                train_ratio=self._train_ratio,
                prefix_data_location=self._prefix_data_location,
            )
            self._train = parser.train
            self._test = parser.test

        return self

    def _set_from_multiple_files(self, infiles: list):
        """Set parameters from multiple input files."""
        self._set_from_single_file(infiles[0])
        if len(infiles) == 1:
            return self

        for infile in infiles[1:]:
            params = ParamsParserSingle(infile, verbose=self._verbose).set_params()
            self._params.append(params)
        return self

    @property
    def priority_file(self):
        """Return priority input file."""
        return self._priority_file

    @property
    def params(self):
        """Return parameters."""
        return self._params

    @property
    def train(self):
        """Return training dataset in DatasetList."""
        return self._train

    @property
    def test(self):
        """Return test dataset in DatasetList."""
        return self._test
