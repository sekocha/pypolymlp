#!/usr/bin/env python
import numpy as np

from pypolymlp.cxx.lib import libmlpcpp


def multiple_dft_dicts_to_mlpcpp_obj(multiple_dft_dicts):

    n_st_dataset, force_dataset = [], []
    axis_array, positions_c_array = [], []
    types_array, n_atoms_sum_array = [], []

    for _, dft_dict in multiple_dft_dicts.items():
        structures = dft_dict["structures"]
        n_st_dataset.append(len(structures))
        force_dataset.append(dft_dict["include_force"])
        res = structures_to_mlpcpp_obj(structures)
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


def structures_to_mlpcpp_obj(structures):

    axis_array = [st["axis"] for st in structures]
    positions_c_array = [np.dot(st["axis"], st["positions"]) for st in structures]
    types_array = [st["types"] for st in structures]
    n_atoms_sum_array = [sum(st["n_atoms"]) for st in structures]
    return (axis_array, positions_c_array, types_array, n_atoms_sum_array)


class Features:

    def __init__(self, params_dict, dft_dict, print_memory=True, element_swap=False):

        if "structures" in dft_dict:
            structures = dft_dict["structures"]
            n_st_dataset = [len(structures)]
            force_dataset = [params_dict["include_force"]]
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
            ) = structures_to_mlpcpp_obj(structures)
        else:
            res = multiple_dft_dicts_to_mlpcpp_obj(dft_dict)
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
                n_st_dataset,
                force_dataset,
            ) = res

        params_dict["element_swap"] = element_swap
        params_dict["print_memory"] = print_memory
        obj = libmlpcpp.PotentialModel(
            params_dict,
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            force_dataset,
            n_atoms_sum_array,
        )
        self.__x = obj.get_x()
        fbegin, sbegin = obj.get_fbegin(), obj.get_sbegin()
        ne, nf, ns = obj.get_n_data()

        ebegin, ei = [], 0
        for n in n_st_dataset:
            ebegin.append(ei)
            ei += n
        ebegin = np.array(ebegin)

        self.__reg_dict = {
            "x": self.__x,
            "first_indices": list(zip(ebegin, fbegin, sbegin)),
            "n_data": (ne, nf, ns),
        }

    @property
    def regression_dict(self):
        return self.__reg_dict

    @property
    def x(self):
        return self.__x

    @property
    def first_indices(self):
        return self.__reg_dict["first_indices"]

    @property
    def n_data(self):
        return self.__reg_dict["n_data"]


class FeaturesHybrid:

    def __init__(
        self,
        hybrid_params_dicts,
        dft_dicts,
        print_memory=True,
        element_swap=False,
    ):
        if "structures" in dft_dicts:
            structures = dft_dicts["structures"]
            n_st_dataset = [len(structures)]
            force_dataset = [hybrid_params_dicts[0]["include_force"]]
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
            ) = structures_to_mlpcpp_obj(structures)
        else:
            res = multiple_dft_dicts_to_mlpcpp_obj(dft_dicts)
            (
                axis_array,
                positions_c_array,
                types_array,
                n_atoms_sum_array,
                n_st_dataset,
                force_dataset,
            ) = res

        for params_dict in hybrid_params_dicts:
            params_dict["element_swap"] = element_swap
            params_dict["print_memory"] = print_memory

        obj = libmlpcpp.PotentialAdditiveModel(
            hybrid_params_dicts,
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            force_dataset,
            n_atoms_sum_array,
        )

        self.__x = obj.get_x()
        fbegin, sbegin = obj.get_fbegin(), obj.get_sbegin()
        cumulative_n_features = obj.get_cumulative_n_features()
        ne, nf, ns = obj.get_n_data()

        ebegin, ei = [], 0
        for n in n_st_dataset:
            ebegin.append(ei)
            ei += n
        ebegin = np.array(ebegin)

        self.__reg_dict = {
            "x": self.__x,
            "first_indices": list(zip(ebegin, fbegin, sbegin)),
            "n_data": (ne, nf, ns),
            "cumulative_n_features": cumulative_n_features,
        }

    @property
    def regression_dict(self):
        return self.__reg_dict

    @property
    def x(self):
        return self.__x

    @property
    def first_indices(self):
        return self.__reg_dict["first_indices"]

    @property
    def n_data(self):
        return self.__reg_dict["n_data"]

    @property
    def cumulative_n_features(self):
        return self.__reg_dict["cumulative_n_features"]
