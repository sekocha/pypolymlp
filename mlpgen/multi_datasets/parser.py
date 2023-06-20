#!/usr/bin/env python
import numpy as np

from pypolymlp.mlpgen.parser import parse_vaspruns

#from pypolymlp.mlpgen.file_parser import ParamsParser

#def parse_params_observations(multiple_infiles):
#
#    multiple_params_dicts = []
#    for infile in args.infile:
#        p = ParamsParser(infile, multiple_datasets=True)
#        params_dict = p.get_params()
#        multiple_params_dicts.append(params_dict)
#    single_params_dict = multiple_params_dicts[0]
#    elements = single_params_dict['elements']
#
#    return single_params_dict, train_dft_dict, test_dft_dict
    
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

   
    
