#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from polymlp_generator.mlpgen.file_parser import parse_vaspruns
from polymlp_generator.mlpgen.file_parser import ParamsParser

from polymlp_generator.mlpgen.multi_datasets.features_sequential \
                                            import FeaturesSequential

from polymlp_generator.mlpgen.regression import Regression
from polymlp_generator.mlpgen.io_potential import save_mlp_lammps

from polymlp_generator.mlpgen.accuracy import compute_error
from polymlp_generator.mlpgen.accuracy import compute_predictions
from polymlp_generator.mlpgen.accuracy import write_error_yaml


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

    t1 = time.time()
    features_train = FeaturesSequential(params_dict, train_dft_dict)
    train_reg_dict = features_train.get_updated_regression_dict()
    features_test = FeaturesSequential(params_dict, 
                                       test_dft_dict,
                                       scales=train_reg_dict['scales'])
    test_reg_dict = features_test.get_updated_regression_dict()

    t2 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, params_dict)
    coeffs, scales = reg.ridge_seq()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(params_dict, coeffs, scales, elements)

    print('  regression: best model')
    print('    alpha: ', mlp_dict['alpha'])

    t3 = time.time()
    error_dict = dict()
    error_dict['train'], error_dict['test'] = dict(), dict()
    for set_id, dft_dict in train_dft_dict.items():
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])
        predictions, weights, indices = compute_predictions(params_dict, 
                                                            dft_dict, 
                                                            coeffs, 
                                                            scales)
        error_dict['train'][set_id] = compute_error(dft_dict, 
                                                    params_dict, 
                                                    predictions, 
                                                    weights,
                                                    indices,
                                                    output_key=output_key)
    for set_id, dft_dict in test_dft_dict.items():
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])
        predictions, weights, indices = compute_predictions(params_dict, 
                                                            dft_dict, 
                                                            coeffs, 
                                                            scales)
        error_dict['test'][set_id] = compute_error(dft_dict, 
                                                   params_dict, 
                                                   predictions, 
                                                   weights,
                                                   indices,
                                                   output_key=output_key)

    write_error_yaml(error_dict['train'])
    write_error_yaml(error_dict['test'], initialize=False)
    t4 = time.time()

    print('  elapsed_time:')
    print('    features + weighting: ', '{:.3f}'.format(t2-t1), '(s)')
    print('    regression:           ', '{:.3f}'.format(t3-t2), '(s)')
    print('    predictions:          ', '{:.3f}'.format(t4-t3), '(s)')


