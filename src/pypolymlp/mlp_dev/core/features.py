"""Class of computing features."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.dataset import DatasetList
from pypolymlp.cxx.lib import libmlpcpp


def _structures_to_mlpcpp_obj(structures: list[PolymlpStructure]):
    """Set structures to apply mlpcpp format."""
    axis_array = [st.axis for st in structures]
    positions_c_array = [st.axis @ st.positions for st in structures]
    types_array = [st.types for st in structures]
    n_atoms_sum_array = [sum(st.n_atoms) for st in structures]
    return (axis_array, positions_c_array, types_array, n_atoms_sum_array)


def _multiple_dft_to_mlpcpp_obj(datasets: DatasetList):
    """Extract structures from multiple datasets."""

    n_st_dataset, force_dataset = [], []
    axis_array, positions_c_array = [], []
    types_array, n_atoms_sum_array = [], []
    for data in datasets:
        n_st_dataset.append(len(data.structures))
        force_dataset.append(data.include_force)
        res = _structures_to_mlpcpp_obj(data.structures)
        axis_array.extend(res[0])
        positions_c_array.extend(res[1])
        types_array.extend(res[2])
        n_atoms_sum_array.extend(res[3])

    return (
        axis_array,
        positions_c_array,
        types_array,
        n_atoms_sum_array,
        n_st_dataset,
        force_dataset,
    )


def _init_features(
    datasets: DatasetList,
    structures: list[PolymlpStructure],
    params: PolymlpParams,
):
    """Initialize structure attributes for passing them to mlpcpp."""
    if datasets is not None:
        (
            axis_array,
            positions_c_array,
            types_array,
            n_atoms_sum_array,
            n_st_dataset,
            force_dataset,
        ) = _multiple_dft_to_mlpcpp_obj(datasets)
    else:
        n_st_dataset = [len(structures)]
        force_dataset = [params.include_force]
        (
            axis_array,
            positions_c_array,
            types_array,
            n_atoms_sum_array,
        ) = _structures_to_mlpcpp_obj(structures)

    return (
        axis_array,
        positions_c_array,
        types_array,
        n_atoms_sum_array,
        n_st_dataset,
        force_dataset,
    )


def _set_ebegin(n_st_dataset: list):
    """Return first indices of datasets for energy entries."""
    ebegin, ei = [], 0
    for n in n_st_dataset:
        ebegin.append(ei)
        ei += n
    return np.array(ebegin)


class Features:
    """Class of computing features."""

    def __init__(
        self,
        params: PolymlpParams,
        datasets: Optional[DatasetList] = None,
        structures: Optional[list[PolymlpStructure]] = None,
        print_memory: bool = True,
        element_swap: bool = False,
    ):
        """Init method."""
        (
            axis_array,
            positions_c_array,
            types_array,
            n_atoms_sum_array,
            n_st_dataset,
            force_dataset,
        ) = _init_features(datasets, structures, params)

        params.element_swap = element_swap
        params.print_memory = print_memory
        obj = libmlpcpp.PotentialModel(
            params.as_dict(),
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            force_dataset,
            n_atoms_sum_array,
        )
        self._x = obj.get_x()
        ebegin = _set_ebegin(n_st_dataset)
        fbegin, sbegin = obj.get_fbegin(), obj.get_sbegin()

        self._first_indices = list(zip(ebegin, fbegin, sbegin))
        self._n_data = tuple(obj.get_n_data())

    @property
    def x(self):
        """Return X."""
        return self._x

    @property
    def first_indices(self):
        """Return first indices of datasets.

        Return
        ------
        Indices in X corresponding to the first data in datasets.
        [
            (ebegin, fbegin, sbegin) for dataset 1,
            (ebegin, fbegin, sbegin) for dataset 2,
            ...
        ]
        """
        return self._first_indices

    @property
    def n_data(self):
        """Return numbers of data entries.

        Return
        ------
        Numbers of data entries. (tuple of ne, nf, ns)
        """
        return self._n_data

    @property
    def cumulative_n_features(self):
        """Return cumulative numbers of features in hybrid models."""
        return None


class FeaturesHybrid:
    """Class of computing features for hybrid model."""

    def __init__(
        self,
        hybrid_params: list[PolymlpParams],
        datasets: Optional[DatasetList] = None,
        structures: Optional[list[PolymlpStructure]] = None,
        print_memory: bool = True,
        element_swap: bool = False,
    ):
        """Init method."""
        (
            axis_array,
            positions_c_array,
            types_array,
            n_atoms_sum_array,
            n_st_dataset,
            force_dataset,
        ) = _init_features(datasets, structures, hybrid_params[0])

        hybrid_params_dicts = []
        for params in hybrid_params:
            params.element_swap = element_swap
            params.print_memory = print_memory
            hybrid_params_dicts.append(params.as_dict())

        obj = libmlpcpp.PotentialHybridModel(
            hybrid_params_dicts,
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            force_dataset,
            n_atoms_sum_array,
        )
        self._x = obj.get_x()
        ebegin = _set_ebegin(n_st_dataset)
        fbegin, sbegin = obj.get_fbegin(), obj.get_sbegin()

        self._first_indices = list(zip(ebegin, fbegin, sbegin))
        self._n_data = tuple(obj.get_n_data())
        self._cumulative_n_features = obj.get_cumulative_n_features()

    @property
    def x(self):
        """Return X."""
        return self._x

    @property
    def first_indices(self):
        """Return first indices of datasets.

        Return
        ------
        Indices in X corresponding to the first data in datasets.
        [
            (ebegin, fbegin, sbegin) for dataset 1,
            (ebegin, fbegin, sbegin) for dataset 2,
            ...
        ]
        """
        return self._first_indices

    @property
    def n_data(self):
        """Return numbers of data entries.

        Return
        ------
        Numbers of data entries. (tuple of ne, nf, ns)
        """
        return self._n_data

    @property
    def cumulative_n_features(self):
        """Return cumulative numbers of features in hybrid models."""
        return self._cumulative_n_features


def compute_features(
    params: Union[PolymlpParams, list[PolymlpParams]],
    datasets: Optional[DatasetList] = None,
    structures: Optional[list[PolymlpStructure]] = None,
    element_swap: bool = False,
    verbose: bool = False,
):
    """Compute polymlp features.

    Parameters
    ----------
    params: Parameters of polymlp.
    datasets: Datasets.
    structures: Structures.
    """
    if isinstance(params, (list, tuple, np.ndarray)):
        features_class = FeaturesHybrid
    else:
        features_class = Features

    features = features_class(
        params,
        datasets=datasets,
        structures=structures,
        print_memory=verbose,
        element_swap=element_swap,
    )
    return features
