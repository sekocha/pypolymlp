#!/usr/bin/env python
import copy

import numpy as np


def print_common_params(params_dict, infile=None):

    if infile is not None:
        print("priority_input:", infile)

    print("common_params:")
    print("  n_types:       ", params_dict["n_type"])
    print("  elements:      ", params_dict["elements"])
    print("  element_order: ", params_dict["element_order"])
    print("  atomic_energy: ", params_dict["atomic_energy"])
    print("  include_force: ", bool(params_dict["include_force"]))
    print("  include_stress:", bool(params_dict["include_stress"]))

    print("  train_data:")
    for v in params_dict["dft"]["train"]:
        print("  -", v)
    print("  test_data:")
    for v in params_dict["dft"]["test"]:
        print("  -", v)


def get_variable_with_max_length(multiple_params_dict, key):

    array = []
    for single in multiple_params_dict:
        if len(single[key]) > len(array):
            array = single[key]
    return array


def set_common_params_dict(multiple_params_dict):

    keys = set()
    for single in multiple_params_dict:
        for k in single.keys():
            keys.add(k)

    common_params_dict = copy.copy(multiple_params_dict[0])

    n_type = max([single["n_type"] for single in multiple_params_dict])

    elements = get_variable_with_max_length(multiple_params_dict, "elements")
    bool_element_order = [
        single["element_order"] for single in multiple_params_dict
    ] == True
    element_order = elements if bool_element_order else None

    atom_e = get_variable_with_max_length(multiple_params_dict, "atomic_energy")

    common_params_dict["n_type"] = n_type
    common_params_dict["elements"] = elements
    common_params_dict["element_order"] = element_order
    common_params_dict["atomic_energy"] = atom_e

    return common_params_dict
