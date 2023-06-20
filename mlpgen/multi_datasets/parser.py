#!/usr/bin/env python
import numpy as np

from pypolymlp.mlpgen.parser import parse_vaspruns

def parse_observations(params_dict):

    elements = params_dict['elements']
    train_dft_dict, test_dft_dict = dict(), dict()
    for set_id, dict1 in params_dict['dft']['train'].items():
        train_dft_dict[set_id] = parse_vaspruns(dict1['vaspruns'],
                                                element_order=elements)
        train_dft_dict[set_id].update(dict1)

    for set_id, dict1 in params_dict['dft']['test'].items():
        test_dft_dict[set_id] = parse_vaspruns(dict1['vaspruns'],
                                               element_order=elements)
        test_dft_dict[set_id].update(dict1)
    return train_dft_dict, test_dft_dict

   
