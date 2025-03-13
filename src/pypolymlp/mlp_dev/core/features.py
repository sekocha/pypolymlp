"""Class of computing features."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import (
    PolymlpDataDFT,
    PolymlpDataXY,
    PolymlpParams,
    PolymlpStructure,
)
from pypolymlp.cxx.lib import libmlpcpp


def _multiple_dft_to_mlpcpp_obj(multiple_dft: list[PolymlpDataDFT]):
    """Extract structures from multiple DFT datasets."""

    n_st_dataset, force_dataset = [], []
    axis_array, positions_c_array = [], []
    types_array, n_atoms_sum_array = [], []
    for dft in multiple_dft:
        n_st_dataset.append(len(dft.structures))
        force_dataset.append(dft.include_force)
        res = _structures_to_mlpcpp_obj(dft.structures)
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


def _structures_to_mlpcpp_obj(structures: PolymlpStructure):
    """Set structures to apply mlpcpp format."""
    axis_array = [st.axis for st in structures]
    positions_c_array = [st.axis @ st.positions for st in structures]
    types_array = [st.types for st in structures]
    n_atoms_sum_array = [sum(st.n_atoms) for st in structures]
    return (axis_array, positions_c_array, types_array, n_atoms_sum_array)


def _init_features(
    dft: Union[PolymlpDataDFT, list[PolymlpDataDFT]],
    structures: list[PolymlpStructure],
    params: PolymlpParams,
):
    """Initialize structure attributes for passing them to mlpcpp."""
    if dft is not None:
        if isinstance(dft, PolymlpDataDFT):
            n_st_dataset = [len(dft.structures)]
            force_dataset = [dft.include_force]
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
            ) = _structures_to_mlpcpp_obj(dft.structures)
        else:
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
                n_st_dataset,
                force_dataset,
            ) = _multiple_dft_to_mlpcpp_obj(dft)
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


def _set_ebegin(n_st_dataset):
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
        dft: Optional[Union[PolymlpDataDFT, list[PolymlpDataDFT]]] = None,
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
        ) = _init_features(dft, structures, params)

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
        ne, nf, ns = obj.get_n_data()

        self._xy = PolymlpDataXY(
            x=self._x,
            first_indices=list(zip(ebegin, fbegin, sbegin)),
            n_data=(ne, nf, ns),
        )

    @property
    def data_xy(self):
        """Return PolymlpDataXY instance where X is assigned."""
        return self._xy

    @property
    def x(self):
        """Return X."""
        return self._x

    @property
    def first_indices(self):
        """Return first indices of datasets."""
        return self._xy.first_indices

    @property
    def n_data(self):
        """Return numbers of data entries."""
        return self._xy.n_data


class FeaturesHybrid:
    """Class of computing features for hybrid model."""

    def __init__(
        self,
        hybrid_params: list[PolymlpParams],
        dft: Optional[Union[PolymlpDataDFT, list[PolymlpDataDFT]]] = None,
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
        ) = _init_features(dft, structures, hybrid_params[0])

        hybrid_params_dicts = []
        for params in hybrid_params:
            params.element_swap = element_swap
            params.print_memory = print_memory
            hybrid_params_dicts.append(params.as_dict())

        obj = libmlpcpp.PotentialAdditiveModel(
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
        cumulative_n_features = obj.get_cumulative_n_features()
        ne, nf, ns = obj.get_n_data()

        self._xy = PolymlpDataXY(
            x=self._x,
            first_indices=list(zip(ebegin, fbegin, sbegin)),
            n_data=(ne, nf, ns),
            cumulative_n_features=cumulative_n_features,
        )

    @property
    def data_xy(self):
        """Return PolymlpDataXY instance where X is assigned."""
        return self._xy

    @property
    def x(self):
        """Return X."""
        return self._x

    @property
    def first_indices(self):
        """Return first indices of datasets."""
        return self._xy.first_indices

    @property
    def n_data(self):
        """Return numbers of data entries."""
        return self._xy.n_data

    @property
    def cumulative_n_features(self):
        """Return cumulative numbers of features in hybrid models."""
        return self._xy.cumulative_n_features
