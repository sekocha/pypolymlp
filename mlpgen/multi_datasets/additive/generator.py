#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from polymlp_generator.mlpgen.file_parser import parse_vaspruns
from polymlp_generator.mlpgen.file_parser import ParamsParser
from polymlp_generator.mlpgen.multi_datasets.additive.features import Features
from polymlp_generator.mlpgen.multi_datasets.precondition import Precondition
from polymlp_generator.mlpgen.regression import Regression
from polymlp_generator.mlpgen.io_potential import save_mlp_lammps

from polymlp_generator.mlpgen.accuracy import compute_error
from polymlp_generator.mlpgen.accuracy import write_error_yaml


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        nargs='*',
                        type=str, 
                        default=['polymlp.in'],
                        help='Input file name')
    args = parser.parse_args()

    multiple_params_dicts = []
    for infile in args.infile:
        p = ParamsParser(infile, multiple_datasets=True)
        params_dict = p.get_params()
        multiple_params_dicts.append(params_dict)
    single_params_dict = multiple_params_dicts[0]
    elements = single_params_dict['elements']

    train_dft_dict, test_dft_dict = dict(), dict()
    for set_id, dict1 in single_params_dict['dft']['train'].items():
        train_dft_dict[set_id] = parse_vaspruns(dict1['vaspruns'], 
                                                element_order=elements)
        train_dft_dict[set_id].update(dict1)

    for set_id, dict1 in single_params_dict['dft']['test'].items():
        test_dft_dict[set_id] = parse_vaspruns(dict1['vaspruns'],
                                               element_order=elements)
        test_dft_dict[set_id].update(dict1)

    t1 = time.time()

    features_train = Features(multiple_params_dicts, train_dft_dict)
    train_reg_dict = features_train.get_regression_dict()

    features_test = Features(multiple_params_dicts, test_dft_dict)
    test_reg_dict = features_test.get_regression_dict()

    t2 = time.time()
    pre_train = Precondition(train_reg_dict, 
                             train_dft_dict, 
                             single_params_dict, 
                             scales=None)
    pre_train.print_data_shape(header='training data size')
    train_reg_dict = pre_train.get_updated_regression_dict()

    pre_test = Precondition(test_reg_dict,
                            test_dft_dict,
                            single_params_dict, 
                            scales=train_reg_dict['scales'])
    pre_test.print_data_shape(header='test data size')
    test_reg_dict = pre_test.get_updated_regression_dict()

    t3 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, single_params_dict)
    coeffs, scales = reg.ridge()
    mlp_dict = reg.get_best_model()

    cumulative_n_features = features_train.get_cumulative_n_features()
    for i, params_dict in enumerate(multiple_params_dicts):
        if i == 0:
            begin, end = 0, cumulative_n_features[0]
        else:
            begin, end = cumulative_n_features[i-1], cumulative_n_features[i]
        save_mlp_lammps(params_dict, 
                        coeffs[begin:end], 
                        scales[begin:end], 
                        elements,
                        filename='polymlp.lammps.'+str(i+1))

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
                                                    single_params_dict, 
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
                                                   single_params_dict, 
                                                   predictions, 
                                                   weights,
                                                   indices,
                                                   output_key=output_key)

    write_error_yaml(error_dict['train'])
    write_error_yaml(error_dict['test'], initialize=False)

    print('  elapsed_time:')
    print('    features:          ', '{:.3f}'.format(t2-t1), '(s)')
    print('    scaling, weighting:', '{:.3f}'.format(t3-t2), '(s)')
    print('    regression:        ', '{:.3f}'.format(t4-t3), '(s)')


