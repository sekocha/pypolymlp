#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from polymlp_generator.common.vasp import parse_vaspruns
from polymlp_generator.mlpgen.params_parser import ParamsParser

from polymlp_generator.mlpgen.multi_datasets.features import Features
from polymlp_generator.mlpgen.multi_datasets.precondition import Precondition
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
            - dataset1
                - vaspruns 
                - include_force
                - weight
                - atomtype
            - dataset2
        - test
            - ...

    Variables in dft_dict (train_dft_dict, test_dft_dict):
        multiple_dft_dicts
        - dataset1
          dft_dict:
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
            - vaspruns 
            - include_force
            - weight
            - atomtype
        - dataset2 ...

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

    p = ParamsParser(args.infile, multiple_datasets=True)
    params_dict = p.get_params()

    train_dft_dict, test_dft_dict = dict(), dict()
    for set_id, dict1 in params_dict['dft']['train'].items():
        train_dft_dict[set_id] = parse_vaspruns(dict1['vaspruns'])
        train_dft_dict[set_id].update(dict1)
        # todo : must be revised
        elements = train_dft_dict[set_id]['elements']

    for set_id, dict1 in params_dict['dft']['test'].items():
        test_dft_dict[set_id] = parse_vaspruns(dict1['vaspruns'])
        test_dft_dict[set_id].update(dict1)

    t1 = time.time()
    train_reg_dict, test_reg_dict = dict(), dict()
    features_train = Features(params_dict, train_dft_dict)
    train_reg_dict['x'] = features_train.get_x()
    train_reg_dict['first_indices'] = features_train.get_first_indices()

    features_test = Features(params_dict, test_dft_dict)
    test_reg_dict['x'] = features_test.get_x()
    test_reg_dict['first_indices'] = features_test.get_first_indices()

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
    save_mlp_lammps(params_dict, coeffs, scales, elements)

    """
    sequential regression
    reg.ridge_seq()
    """
    print('  regression: best model')
    print('    alpha: ', mlp_dict['alpha'])

    t4 = time.time()
    error_dict = dict()
    error_dict['train'], error_dict['test'] = dict(), dict()
    for (set_id, dft_dict), indices in zip(train_dft_dict.items(), 
                                           train_reg_dict['first_indices']):
        predictions = mlp_dict['predictions']['train']
        weights = train_reg_dict['weight']
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])
        error_dict['train'][set_id] = compute_error(dft_dict, 
                                                    params_dict, 
                                                    predictions, 
                                                    weights,
                                                    indices,
                                                    output_key=output_key)

    for (set_id, dft_dict), indices in zip(test_dft_dict.items(), 
                                           test_reg_dict['first_indices']):
        predictions = mlp_dict['predictions']['test']
        weights = test_reg_dict['weight']
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])
        error_dict['test'][set_id] = compute_error(dft_dict, 
                                                   params_dict, 
                                                   predictions, 
                                                   weights,
                                                   indices,
                                                   output_key=output_key)
    write_error_yaml(error_dict)

    print('  elapsed_time:')
    print('    features:          ', '{:.3f}'.format(t2-t1), '(s)')
    print('    scaling, weighting:', '{:.3f}'.format(t3-t2), '(s)')
    print('    regression:        ', '{:.3f}'.format(t4-t3), '(s)')

    """ 
    sequential regression error
    error = EstimatePredictionErrorFromPot(data_train, data_test, pot_e)
    """

