"""Class of input parameter parser."""

import copy
from typing import Literal, Optional, Union

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParams
from pypolymlp.core.dataset import Dataset
from pypolymlp.core.parser_infile import InputParser
from pypolymlp.core.polymlp_params import (
    set_active_gaussian_params,
    set_gaussian_params,
    set_gtinv_params,
    set_regression_alphas,
)


def set_data_locations(
    parser: InputParser,
    include_force: bool = True,
    train_ratio: float = 0.9,
):
    """Set locations of data for multiple datasets."""
    dft_train, dft_test = [], []
    for dataset in parser.train:
        if not include_force:
            dataset.include_force = False
        dft_train.append(dataset)

    for dataset in parser.test:
        if not include_force:
            dataset.include_force = False
        dft_test.append(dataset)

    for dataset in parser.train_test:
        if not include_force:
            dataset.include_force = False
        train, test = dataset.split_train_test(train_ratio=train_ratio)
        dft_train.append(train)
        dft_test.append(test)

    for dataset in parser.md:
        if not include_force:
            dataset.include_force = False
        dataset.split = False
        dft_train.append(dataset)

    return dft_train, dft_test


class ParamsParser:
    """Class of input parameter parser."""

    def __init__(
        self,
        filename: str,
        parse_vasprun_locations: bool = True,
        prefix: Optional[str] = None,
        train_ratio: float = 0.9,
    ):
        """Init class.

        Parameters
        ----------
        filename: File of input parameters for single polymlp (e.g., polymlp.in).
        """
        self.parser = InputParser(filename, prefix=prefix)
        include_force, include_stress = self._set_force_tags()
        self.include_force = include_force
        self._train_ratio = train_ratio

        elements, n_type, atomic_energy = self._set_element_properties()
        self._elements = elements
        rearrange = self.parser.get_params(
            "rearrange_by_elements",
            default=True,
            dtype=bool,
        )
        element_order = elements if rearrange else None
        alphas = self._get_regression_params()
        model = self._get_potential_model_params(n_type)

        if parse_vasprun_locations:
            dataset_type = self.parser.get_params("dataset_type", default="vasp")
            dft_train, dft_test = self._get_dataset(dataset_type)
        else:
            dataset_type = "vasp"
            dft_train, dft_test = None, None

        self._params = PolymlpParams(
            n_type=n_type,
            elements=elements,
            model=model,
            atomic_energy=atomic_energy,
            dft_train=dft_train,
            dft_test=dft_test,
            regression_alpha=alphas,
            include_force=include_force,
            include_stress=include_stress,
            dataset_type=dataset_type,
            element_order=element_order,
        )

        if dataset_type == "electron":
            self._params.temperature = self.parser.get_params(
                "temperature", default=300, dtype=float
            )
            self._params.electron_property = self.parser.get_params(
                "electron_property", default="free_energy", dtype=str
            )

    def _set_force_tags(self):
        """Set include_force and include_stress."""
        include_force = self.parser.get_params(
            "include_force",
            default=True,
            dtype=bool,
        )
        if include_force:
            include_stress = self.parser.get_params(
                "include_stress", default=True, dtype=bool
            )
        else:
            include_stress = False
        return include_force, include_stress

    def _set_element_properties(self):
        """Set properties for identifying elements."""
        n_type = self.parser.get_params("n_type", default=1, dtype=int)
        elements = self.parser.get_params(
            "elements",
            size=n_type,
            default=None,
            required=True,
            dtype=str,
            return_array=True,
        )
        d_atom_e = [0.0 for i in range(n_type)]
        atom_e = self.parser.get_params(
            "atomic_energy",
            size=n_type,
            default=d_atom_e,
            dtype=float,
            return_array=True,
        )
        return elements, n_type, tuple(atom_e)

    def _get_regression_params(self):
        """Set regularization parameters in regression."""
        alpha_params = self.parser.get_params(
            "reg_alpha_params",
            size=3,
            default=(-3.0, 1.0, 5),
            dtype=float,
            return_array=True,
        )
        alphas = set_regression_alphas(alpha_params)
        return alphas

    def _get_gtinv_params(self, n_type: int, feature_type: Literal["gtinv", "pair"]):
        """Set parameters for group-theoretical invariants."""
        version = self.parser.get_params("gtinv_version", default=1, dtype=int)
        order = self.parser.get_params("gtinv_order", default=3, dtype=int)
        size = order - 1
        gtinv_maxl = self.parser.get_params(
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

    def _get_potential_model_params(self, n_type: int):
        """Set parameters for identifying potential model."""
        cutoff = self.parser.get_params("cutoff", default=6.0, dtype=float)
        model_type = self.parser.get_params("model_type", default=1, dtype=int)
        max_p = self.parser.get_params("max_p", default=1, dtype=int)
        feature_type = self.parser.get_params("feature_type", default="gtinv")

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

    def _get_pair_params(self, cutoff: float):
        """Set parameters for Gaussian radial functions."""
        params1 = self.parser.get_params(
            "gaussian_params1",
            size=3,
            default=(1.0, 1.0, 1),
            dtype=float,
            return_array=True,
        )
        params2 = self.parser.get_params(
            "gaussian_params2",
            size=3,
            default=(0.0, cutoff - 1.0, 7),
            dtype=float,
            return_array=True,
        )
        distance = self.parser.distance
        pair_params = set_gaussian_params(params1, params2)
        pair_params_active, cond = set_active_gaussian_params(
            pair_params,
            self._elements,
            distance,
        )
        return pair_params, pair_params_active, cond

    def _get_dataset(
        self,
        dataset_type: Literal["vasp", "phono3py", "sscha", "electron"],
    ):
        """Set files in datasets."""
        if dataset_type in ["vasp", "sscha", "electron"]:
            if len(self.parser.train) == 0 and len(self.parser.train_test) == 0:
                raise RuntimeError("Training data not found.")
            if len(self.parser.test) == 0 and len(self.parser.train_test) == 0:
                raise RuntimeError("Test data not found.")

            self._multiple_datasets = True
            dft_train, dft_test = set_data_locations(
                self.parser,
                include_force=self.include_force,
                train_ratio=self._train_ratio,
            )
            return dft_train, dft_test
        elif dataset_type == "phono3py":
            self._multiple_datasets = False
            return self._get_phono3py_set()
        else:
            raise KeyError("Given dataset_type is unavailable.")

    def _get_phono3py_set(self):
        """Set dataset for input in phono3py format."""
        yml = self.parser.get_params("data_phono3py", size=2, default=None)
        location = yml[0]
        energy_dat = yml[1] if len(yml) > 1 else None

        dft_train = Dataset(
            dataset_type="phono3py",
            name=location,
            location=location,
            include_force=self.include_force,
            split=False,
            energy_dat=energy_dat,
        )
        dft_train, dft_test = [dft_train], []
        return dft_train, dft_test

    @property
    def params(self) -> PolymlpModelParams:
        """Return parameters for developing polymlp."""
        return self._params


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
    common_params = None
    hybrid_params = None
    priority_infile = None
    is_hybrid = False
    if not isinstance(infiles, list):
        p = ParamsParser(infiles, prefix=prefix)
        common_params = p.params
        priority_infile = infiles
    else:
        priority_infile = infiles[0]
        if len(infiles) == 1:
            p = ParamsParser(priority_infile, prefix=prefix)
            common_params = p.params
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
            is_hybrid = True

    return (common_params, hybrid_params, is_hybrid, priority_infile)
