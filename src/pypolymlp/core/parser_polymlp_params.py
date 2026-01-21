"""Class of input parameter parser."""

import copy
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParams
from pypolymlp.core.dataset import Dataset, DatasetList
from pypolymlp.core.parser_infile import InputParser
from pypolymlp.core.polymlp_params import (
    set_active_gaussian_params,
    set_gaussian_params,
    set_gtinv_params,
    set_regression_alphas,
)


class ParamsParser:
    """Class of input parameter parser."""

    def __init__(self, filename: str, verbose: bool = False):
        """Init method.

        Parameters
        ----------
        filename: File of input parameters for single polymlp (e.g., polymlp.in).
        """
        self._parser = InputParser(filename, verbose=verbose)
        self._verbose = verbose

        self._include_force = None
        self._elements = None
        self._params = None

        self._train = None
        self._test = None

    def set_params(self):
        """Get parameters from file and set them."""
        include_force, include_stress = self._get_force_tags()
        elements, n_type, atom_e = self._get_element_properties()

        rearrange = self.parser.get_params(
            "rearrange_by_elements", default=True, dtype=bool
        )
        element_order = elements if rearrange else None
        dataset_type = self._parser.get_params("dataset_type", default="vasp")

        alphas = self._get_regression_params()
        model = self._get_potential_model_params(n_type)

        self._params = PolymlpParams(
            n_type=n_type,
            elements=elements,
            model=model,
            atomic_energy=atom_e,
            regression_alpha=alphas,
            include_force=include_force,
            include_stress=include_stress,
            dataset_type=dataset_type,
            element_order=element_order,
        )

        if dataset_type == "electron":
            self._params.temperature = self._parser.get_params(
                "temperature", default=300, dtype=float
            )
            self._params.electron_property = self._parser.get_params(
                "electron_property", default="free_energy", dtype=str
            )
        return self

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
        self._elements = elements
        return elements, n_type, tuple(atom_e)

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

    def _get_potential_model_params(self, n_type: int):
        """Set parameters for identifying potential model."""
        cutoff = self._parser.get_params("cutoff", default=6.0, dtype=float)
        model_type = self._parser.get_params("model_type", default=1, dtype=int)
        max_p = self._parser.get_params("max_p", default=1, dtype=int)
        feature_type = self._parser.get_params("feature_type", default="gtinv")

        gtinv_params, max_l = self._get_gtinv_params(n_type, feature_type)
        pair_params, pair_params_active, pair_cond = self._get_pair_params(cutoff)

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
        distance = self._parser.distance
        pair_params = set_gaussian_params(params1, params2)
        pair_params_active, cond = set_active_gaussian_params(
            pair_params,
            self._elements,
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
            if not self._include_force:
                dataset.include_force = False

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
                dataset.split = False
                train_set.append(dataset)

        self._train = DatasetList(train_set)
        self._test = DatasetList(test_set)
        not_split_data = DatasetList(not_split_set)

        self._train.parse_files(self._params)
        self._test.parse_files(self._params)
        not_split_data.parse_files(self._params)

        for dataset in not_split_data:
            train, test = dataset.split_dft(train_ratio=train_ratio)
            self._train.append(train)
            self._test.append(test)

        return self

    @property
    def params(self) -> PolymlpModelParams:
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


def _get_variable_with_max_length(
    multiple_params: list[PolymlpParams], key: str
) -> list:
    """Select variable with max length."""
    array = []
    for single in multiple_params:
        single_dict = single.as_dict()
        if len(single_dict[key]) > len(array):
            array = single_dict[key]
    return array


def set_common_params(multiple_params: list[PolymlpParams]) -> PolymlpParams:
    """Set common parameters of multiple PolymlpParams."""
    # TODO: DFT datasets should be automatically found
    # if they are not included in the first file.
    keys = set()
    for single in multiple_params:
        for k in single.as_dict().keys():
            keys.add(k)

    common_params = copy.copy(multiple_params[0])
    n_type = max([single.n_type for single in multiple_params])
    elements = _get_variable_with_max_length(multiple_params, "elements")
    atom_e = _get_variable_with_max_length(multiple_params, "atomic_energy")

    bool_element_order = [
        single.element_order for single in multiple_params
    ] is not None
    element_order = elements if bool_element_order else None

    common_params.n_type = n_type
    common_params.elements = elements
    common_params.element_order = element_order
    common_params.atomic_energy = atom_e
    return common_params


def _set_unique_types(
    multiple_params: list[PolymlpParams],
    common_params: PolymlpParams,
):
    """Set type indices for hybrid models."""
    n_type = common_params.n_type
    elements = common_params.elements
    for single in multiple_params:
        single.elements = sorted(single.elements, key=lambda x: elements.index(x))

    for single in multiple_params:
        if single.n_type == n_type:
            single.type_full = True
            single.type_indices = list(range(n_type))
        else:
            single.type_full = False
            single.type_indices = [elements.index(ele) for ele in single.elements]
    return multiple_params


def parse_parameter_files(infiles: Union[str, list[str]], prefix: str = None):
    """Parse input files for developing polymlp."""
    if isinstance(infiles, (list, tuple, np.ndarray)):
        priority_infile = infiles[0]
        if len(infiles) == 1:
            p = ParamsParser(priority_infile, prefix=prefix)
            params = common_params = p.params
            hybrid_params = None
        else:
            hybrid_params = []
            for i, infile in enumerate(infiles):
                if i == 0:
                    params = ParamsParser(infile, prefix=prefix).params
                else:
                    params = ParamsParser(
                        infile, parse_vasprun_locations=False, prefix=prefix
                    ).params
                hybrid_params.append(params)
            common_params = set_common_params(hybrid_params)
            hybrid_params = _set_unique_types(hybrid_params, common_params)
            params = hybrid_params
    else:
        priority_infile = infiles
        p = ParamsParser(infiles, prefix=prefix)
        params = common_params = p.params
        hybrid_params = None

    common_params.priority_infile = priority_infile
    return (params, common_params, hybrid_params)
