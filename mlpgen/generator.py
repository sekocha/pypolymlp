#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.common.interface_phono3py import parse_phono3py_yaml
from pypolymlp.mlpgen.parser import parse_vaspruns
from pypolymlp.mlpgen.parser import ParamsParser
from pypolymlp.mlpgen.features import Features
from pypolymlp.mlpgen.precondition import Precondition
from pypolymlp.mlpgen.regression import Regression
from pypolymlp.mlpgen.io_potential import save_mlp_lammps

from pypolymlp.mlpgen.accuracy import compute_error
from pypolymlp.mlpgen.accuracy import write_error_yaml
from pypolymlp.mlpgen.features_attr import write_polymlp_params_yaml


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
        - train (vasprun locations)
        - test (vasprun locations)

    Variables in dft_dict (train_dft_dict, test_dft_dict):
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
        - elements
        - total_n_atoms

    Variables in reg_dict
        - x
        - y
        - first_indices [(ebegin, fbegin, sbegin), ...]
        - n_data (ne, nf, ns)
        - scales

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

    if params_dict['dataset_type'] == 'vasp':
        train_dft_dict = parse_vaspruns(params_dict['dft']['train'],
                                    element_order=params_dict['element_order'])
        test_dft_dict = parse_vaspruns(params_dict['dft']['test'],
                                    element_order=params_dict['element_order'])
    elif params_dict['dataset_type'] == 'phono3py':
        train_dft_dict = parse_phono3py_yaml(
            params_dict['dft']['train']['phono3py_yaml'],
            params_dict['dft']['train']['energy'],
            element_order=params_dict['element_order'],
            select_ids=params_dict['dft']['train']['indices']
        )
        test_dft_dict = parse_phono3py_yaml(
            params_dict['dft']['test']['phono3py_yaml'],
            params_dict['dft']['test']['energy'],
            element_order=params_dict['element_order'],
            select_ids=params_dict['dft']['test']['indices']
        )


    t1 = time.time()
    features_train = Features(params_dict, train_dft_dict['structures'])
    train_reg_dict = features_train.get_regression_dict()
    features_test = Features(params_dict, test_dft_dict['structures'])
    test_reg_dict = features_test.get_regression_dict()

    t2 = time.time()
    pre_train = Precondition(train_reg_dict, 
                             train_dft_dict, 
                             params_dict, 
                             scales=None)
    pre_train.print_data_shape(header='training data size')
    train_reg_dict = pre_train.get_updated_regression_dict()

    pre_test = Precondition(test_reg_dict, 
                            test_dft_dict, 
                            params_dict, 
                            scales=train_reg_dict['scales'])
    pre_test.print_data_shape(header='test data size')
    test_reg_dict = pre_test.get_updated_regression_dict()

    t3 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, params_dict)
    coeffs, scales = reg.ridge()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(params_dict, coeffs, scales)

    print('  regression: best model')
    print('    alpha: ', mlp_dict['alpha'])

    t4 = time.time()
    error_dict = dict()
    indices = train_reg_dict['first_indices'][0]
    error_dict['train'] = compute_error(train_dft_dict, 
                                        params_dict, 
                                        mlp_dict['predictions']['train'],
                                        train_reg_dict['weight'],
                                        indices, 
                                        output_key='train')
    indices = test_reg_dict['first_indices'][0]
    error_dict['test'] = compute_error(test_dft_dict, 
                                       params_dict, 
                                       mlp_dict['predictions']['test'],
                                       test_reg_dict['weight'],
                                       indices,
                                       output_key='test')
    write_error_yaml(error_dict)
    write_polymlp_params_yaml(params_dict)

    print('  elapsed_time:')
    print('    features:          ', '{:.3f}'.format(t2-t1), '(s)')
    print('    scaling, weighting:', '{:.3f}'.format(t3-t2), '(s)')
    print('    regression:        ', '{:.3f}'.format(t4-t3), '(s)')

