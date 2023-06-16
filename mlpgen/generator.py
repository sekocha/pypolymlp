#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from polymlp_generator.common.vasp import parse_vaspruns
from polymlp_generator.mlpgen.params_parser import ParamsParser
from polymlp_generator.mlpgen.features import Features
from polymlp_generator.mlpgen.precondition import Precondition
from polymlp_generator.mlpgen.regression import Regression
from polymlp_generator.mlpgen.io_potential import save_mlp_lammps

from polymlp_generator.mlpgen.accuracy import compute_error
from polymlp_generator.mlpgen.accuracy import write_error_yaml


"""
    Variables in params_dict:
      - n_type
      - include_force
      - include_stress
      - model
        - cutoff
        - model_type
        - max_p
        - max_l
        - feature_type
        - pair_type
        - pair_params
        - gtinv
          - order
          - max_l
          - lm_seq
          - l_comb
          - lm_coeffs
      - atomic_energy
      - reg
        - method
        - alpha
      - dft
        - train
        - test

    Variables in dft_dict:
      - train
        - energy
        - force
        - stress
        - structures
          - structure (1) 
            - axis
            - positions
            - n_atoms
            - types
            - elements
          - ...
      - test
        - ...

    Variables in reg_dict
      - train
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
      - test
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
      - scaler

"""

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        type=str, 
                        default='polymlp.in',
                        help='Input file name')
    args = parser.parse_args()

    p = ParamsParser(args.infile)
    params_dict = p.get_params()

    dft_dict = dict()
    dft_dict['train'] = parse_vaspruns(params_dict['dft']['train'])
    dft_dict['test'] = parse_vaspruns(params_dict['dft']['test'])
    elements = dft_dict['train']['elements']

    reg_dict = dict()
    reg_dict['train'], reg_dict['test'] = dict(), dict()

    features_train = Features(params_dict, dft_dict['train']['structures'])
    reg_dict['train']['x'] = features_train.get_x()
    reg_dict['train']['first_indices'] = features_train.get_first_indices()

    features_test = Features(params_dict, dft_dict['test']['structures'])
    reg_dict['test']['x'] = features_test.get_x()
    reg_dict['test']['first_indices'] = features_test.get_first_indices()

    pre_train = Precondition(reg_dict['train'], 
                             dft_dict['train'], 
                             params_dict, 
                             scaler=None)
    pre_train.print_data_shape(header='training data size')
    reg_dict['scaler'] = pre_train.get_scaler()

    pre_test = Precondition(reg_dict['test'], 
                            dft_dict['test'], 
                            params_dict, 
                            scaler=reg_dict['scaler'])
    pre_test.print_data_shape(header='test data size')

    reg = Regression(reg_dict, params_dict)
    coeffs, scales = reg.ridge()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(params_dict, coeffs, scales, elements)

    """
    pot_e = reg.ridge_seq(alpha_min=alpha_min,
                          alpha_max=alpha_max,
                          n_alpha=n_alpha)
    """
    
    print('  regression: best model')
    print('    alpha: ', mlp_dict['alpha'])

    error_dict = dict()
    error_dict['train'] = compute_error(reg_dict, 
                                        dft_dict, 
                                        params_dict, 
                                        mlp_dict, 
                                        key='train')
    error_dict['test'] = compute_error(reg_dict, 
                                       dft_dict, 
                                       params_dict, 
                                       mlp_dict, 
                                       key='test')

    write_error_yaml(error_dict)
    """
    print yaml file.
    """

#    print(' elapsed time (electrostatic)   =', '{:.3f}'.format(t2-t1), '(s)')
#    print(' elapsed time (features)        =', '{:.3f}'.format(t3-t2), '(s)')
#    print(' elapsed time (scaling, weight) =', '{:.3f}'.format(t4-t3), '(s)')
#    print(' elapsed time (regression)      =', '{:.3f}'.format(t5-t4), '(s)')
#    print(' elapsed time (prediction)      =', '{:.3f}'.format(t6-t5), '(s)')
#    print(' elapsed time (print files)     =', '{:.3f}'.format(t7-t6), '(s)')

""" 
    seq. error
#            error = EstimatePredictionErrorFromPot(data_train, 
#                                                   data_test,
#                                                   pot_e)
"""

