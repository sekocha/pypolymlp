"""Class of computing features"""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import (
    PolymlpDataDFT,
    PolymlpDataXY,
    PolymlpParams,
    PolymlpStructure,
)
from pypolymlp.cxx.lib import libmlpcpp


def multiple_dft_to_mlpcpp_obj(multiple_dft: list[PolymlpDataDFT]):

    n_st_dataset, force_dataset = [], []
    axis_array, positions_c_array = [], []
    types_array, n_atoms_sum_array = [], []

    for dft in multiple_dft:
        n_st_dataset.append(len(dft.structures))
        force_dataset.append(dft.include_force)
        res = structures_to_mlpcpp_obj(dft.structures)
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


def structures_to_mlpcpp_obj(structures: PolymlpStructure):

    axis_array = [st.axis for st in structures]
    positions_c_array = [st.axis @ st.positions for st in structures]
    types_array = [st.types for st in structures]
    n_atoms_sum_array = [sum(st.n_atoms) for st in structures]
    return (axis_array, positions_c_array, types_array, n_atoms_sum_array)


class Features:

    def __init__(
        self,
        params: PolymlpParams,
        dft: Optional[Union[PolymlpDataDFT, list[PolymlpDataDFT]]] = None,
        structures: Optional[list[PolymlpStructure]] = None,
        print_memory: bool = True,
        element_swap: bool = False,
    ):

        if dft is not None:
            if isinstance(dft, PolymlpDataDFT):
                n_st_dataset = [len(dft.structures)]
                force_dataset = [dft.include_force]
                (
                    axis_array,
                    positions_c_array,
                    types_array,
                    n_atoms_sum_array,
                ) = structures_to_mlpcpp_obj(dft.structures)
            else:
                (
                    axis_array,
                    positions_c_array,
                    types_array,
                    n_atoms_sum_array,
                    n_st_dataset,
                    force_dataset,
                ) = multiple_dft_to_mlpcpp_obj(dft)
        else:
            n_st_dataset = [len(structures)]
            force_dataset = [params.include_force]
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
            ) = structures_to_mlpcpp_obj(structures)

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
        fbegin, sbegin = obj.get_fbegin(), obj.get_sbegin()
        ne, nf, ns = obj.get_n_data()

        ebegin, ei = [], 0
        for n in n_st_dataset:
            ebegin.append(ei)
            ei += n
        ebegin = np.array(ebegin)

        self._xy = PolymlpDataXY(
            x=self._x,
            first_indices=list(zip(ebegin, fbegin, sbegin)),
            n_data=(ne, nf, ns),
        )

    @property
    def data_xy(self):
        return self._xy

    @property
    def x(self):
        return self._x

    @property
    def first_indices(self):
        return self._xy.first_indices

    @property
    def n_data(self):
        return self._xy.n_data


class FeaturesHybrid:

    def __init__(
        self,
        hybrid_params: list[PolymlpParams],
        dft: Optional[Union[PolymlpDataDFT, list[PolymlpDataDFT]]] = None,
        structures: Optional[list[PolymlpStructure]] = None,
        print_memory: bool = True,
        element_swap: bool = False,
    ):
        if dft is not None:
            if isinstance(dft, PolymlpDataDFT):
                n_st_dataset = [len(dft.structures)]
                force_dataset = [dft.include_force]
                (
                    axis_array,
                    positions_c_array,
                    types_array,
                    n_atoms_sum_array,
                ) = structures_to_mlpcpp_obj(dft.structures)
            else:
                (
                    axis_array,
                    positions_c_array,
                    types_array,
                    n_atoms_sum_array,
                    n_st_dataset,
                    force_dataset,
                ) = multiple_dft_to_mlpcpp_obj(dft)
        else:
            n_st_dataset = [len(structures)]
            force_dataset = [hybrid_params[0].include_force]
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
            ) = structures_to_mlpcpp_obj(structures)

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
        fbegin, sbegin = obj.get_fbegin(), obj.get_sbegin()
        cumulative_n_features = obj.get_cumulative_n_features()
        ne, nf, ns = obj.get_n_data()

        ebegin, ei = [], 0
        for n in n_st_dataset:
            ebegin.append(ei)
            ei += n
        ebegin = np.array(ebegin)

        self._xy = PolymlpDataXY(
            x=self._x,
            first_indices=list(zip(ebegin, fbegin, sbegin)),
            n_data=(ne, nf, ns),
            cumulative_n_features=cumulative_n_features,
        )

    @property
    def data_xy(self):
        return self._xy

    @property
    def x(self):
        return self._x

    @property
    def first_indices(self):
        return self._xy.first_indices

    @property
    def n_data(self):
        return self._xy.n_data

    @property
    def cumulative_n_features(self):
        return self._xy.cumulative_n_features
