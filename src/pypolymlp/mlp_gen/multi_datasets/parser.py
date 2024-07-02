#!/usr/bin/env python
import numpy as np

from pypolymlp.core.interface_vasp import parse_vaspruns


def parse_observations(params_dict):

    element_order = params_dict["element_order"]
    train_dft_dict, test_dft_dict = dict(), dict()
    for set_id, dict1 in params_dict["dft"]["train"].items():
        train_dft_dict[set_id] = parse_vaspruns(
            dict1["vaspruns"], element_order=element_order
        )
        train_dft_dict[set_id].update(dict1)

    for set_id, dict1 in params_dict["dft"]["test"].items():
        test_dft_dict[set_id] = parse_vaspruns(
            dict1["vaspruns"], element_order=element_order
        )
        test_dft_dict[set_id].update(dict1)

    return train_dft_dict, test_dft_dict
