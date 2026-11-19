"""Class of input parameter parser."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.dataset import Dataset, DatasetList
from pypolymlp.core.params import PolymlpParams
from pypolymlp.core.params_utils import set_all_params
from pypolymlp.core.parser_infile import InputParser


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

        self._include_force = None

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
        elements, n_type, atom_e, unit, enable_spins = self._get_element_properties()
        alpha_params, use_grad, use_cv = self._get_solver_params()
        batch_size, gtol, max_iter = self._get_solver_options()
        use_weight, weight_s, weight_method, weight_min_e = self._get_weight_params()
        solver_method = "ridge"

        cutoff, model_type, max_p, feature_type = self._get_potential_model_params()
        gparams1, gparams2, n_gaussians, distance = self._get_pair_params(cutoff)
        gtinv_order, gtinv_maxl, gtinv_version = self._get_gtinv_params()
        self._params = set_all_params(
            elements=elements,
            cutoff=cutoff,
            model_type=model_type,
            max_p=max_p,
            feature_type=feature_type,
            gaussian_params1=gparams1,
            gaussian_params2=gparams2,
            n_gaussians=n_gaussians,
            distance=distance,
            gtinv_order=gtinv_order,
            gtinv_maxl=gtinv_maxl,
            gtinv_version=gtinv_version,
            include_force=include_force,
            include_stress=include_stress,
            atomic_energy_unit=unit,
            atomic_energy=atom_e,
            enable_spins=enable_spins,
            solver_method=solver_method,
            solver_alpha_params=alpha_params,
            solver_use_gradient=use_grad,
            solver_use_cv=use_cv,
            solver_batch_size=batch_size,
            solver_gtol=gtol,
            solver_max_iter=max_iter,
            use_auto_weight=use_weight,
            weight_stress=weight_s,
            weight_method=weight_method,
            weight_min_energy=weight_min_e,
        )

        dataset_type = self._parser.get_params("dataset_type", default="vasp")
        self._params.dataset_type = dataset_type
        if dataset_type == "electron":
            temp, prop_name = self._get_electron_tags()
            self._params.temperature = temp
            self._params.electron_property = prop_name

        return self._params

    def _get_electron_tags(self):
        """Return tags for electronic free energy model."""
        temp = self._parser.get_params("temperature", default=300, dtype=float)
        prop_name = self._parser.get_params(
            "electron_property", default="free_energy", dtype=str
        )
        return (temp, prop_name)

    def _get_force_tags(self):
        """Return include_force and include_stress."""
        inc_force = self._parser.get_params("include_force", default=True, dtype=bool)
        if inc_force:
            inc_stress = self._parser.get_params(
                "include_stress", default=True, dtype=bool
            )
        else:
            inc_stress = False

        self._include_force = inc_force
        return (inc_force, inc_stress)

    def _get_element_properties(self):
        """Return properties for identifying elements."""
        n_type = self._parser.get_params("n_type", required=True, dtype=int)
        elements = self._parser.get_params(
            "elements",
            size=n_type,
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
        enable_spins = self._parser.get_params(
            "enable_spins",
            size=n_type,
            default=None,
            dtype=bool,
            return_array=True,
        )
        return (elements, n_type, tuple(atom_e), atom_e_unit, enable_spins)

    def _get_solver_params(self):
        """Set parameters in solver."""
        alpha_params = self._parser.get_params(
            "reg_alpha_params",
            size=3,
            default=None,
            dtype=float,
            return_array=True,
        )
        if alpha_params is None:
            alpha_params = self._parser.get_params(
                "solver_alpha_params",
                size=3,
                default=(-3.0, 1.0, 5),
                dtype=float,
                return_array=True,
            )

        use_gradient = self._parser.get_params(
            "solver_use_gradient",
            default=False,
            dtype=bool,
        )
        use_cv = self._parser.get_params(
            "solver_use_cv",
            default=False,
            dtype=bool,
        )
        return (alpha_params, use_gradient, use_cv)

    def _get_solver_options(self):
        """Set optional parameters in solver."""
        batch_size = self._parser.get_params(
            "solver_batch_size",
            default=None,
            dtype=int,
        )
        gtol = self._parser.get_params("solver_gtol", default=1e-2, dtype=float)
        max_iter = self._parser.get_params("solver_max_iter", default=None, dtype=int)
        return (batch_size, gtol, max_iter)

    def _get_weight_params(self):
        """Set parameters for weights."""
        use_weight = self._parser.get_params(
            "use_auto_weight",
            default=True,
            dtype=bool,
        )
        weight_stress = self._parser.get_params(
            "weight_stress",
            default=0.1,
            dtype=float,
        )
        weight_method = self._parser.get_params(
            "weight_method",
            default=1,
            dtype=int,
        )
        weight_min_energy = self._parser.get_params(
            "weight_min_energy",
            default=None,
            dtype=float,
        )
        return (use_weight, weight_stress, weight_method, weight_min_energy)

    def _get_potential_model_params(self):
        """Set parameters for identifying potential model."""
        cutoff = self._parser.get_params("cutoff", default=6.0, dtype=float)
        model_type = self._parser.get_params("model_type", default=1, dtype=int)
        max_p = self._parser.get_params("max_p", default=1, dtype=int)
        feature_type = self._parser.get_params("feature_type", default="gtinv")
        return (cutoff, model_type, max_p, feature_type)

    def _get_pair_params(self, cutoff: float):
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
        return (params1, params2, n_gaussians, distance)

    def _get_gtinv_params(self):
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

        return (order, gtinv_maxl, version)

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
