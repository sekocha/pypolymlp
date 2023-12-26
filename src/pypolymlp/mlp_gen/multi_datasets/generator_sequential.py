#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.core.io_polymlp import save_mlp_lammps

from pypolymlp.mlp_gen.multi_datasets.parser import parse_observations
from pypolymlp.mlp_gen.multi_datasets.sequential import Sequential
from pypolymlp.mlp_gen.regression import Regression

from pypolymlp.mlp_gen.accuracy import compute_error
from pypolymlp.mlp_gen.multi_datasets.accuracy import compute_predictions
from pypolymlp.mlp_gen.accuracy import write_error_yaml
from pypolymlp.mlp_gen.features_attr import write_polymlp_params_yaml

def run_sequential_generator_multiple_datasets(infile):

    p = ParamsParser(infile, multiple_datasets=True)
    params_dict = p.get_params()

    train_dft_dict, test_dft_dict = parse_observations(params_dict)

    t1 = time.time()
    seq_train = Sequential(params_dict, train_dft_dict)
    train_reg_dict = seq_train.get_updated_regression_dict()
    seq_test = Sequential(params_dict, 
                          test_dft_dict,
                          scales=train_reg_dict['scales'])
    test_reg_dict = seq_test.get_updated_regression_dict()

    t2 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, params_dict)
    coeffs, scales = reg.ridge_seq()
    mlp_dict = reg.get_best_model()
    save_mlp_lammps(params_dict, coeffs, scales)

    print('  regression: best model')
    print('    alpha: ', mlp_dict['alpha'])

    t3 = time.time()
    error_dict = dict()
    error_dict['train'], error_dict['test'] = dict(), dict()
    for set_id, dft_dict in train_dft_dict.items():
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])\
                        .replace('..','')
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
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])\
                        .replace('..','')
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
    t4 = time.time()

    write_error_yaml(error_dict['train'])
    write_error_yaml(error_dict['test'], initialize=False)
    write_polymlp_params_yaml(params_dict)

    print('  elapsed_time:')
    print('    features + weighting: ', '{:.3f}'.format(t2-t1), '(s)')
    print('    regression:           ', '{:.3f}'.format(t3-t2), '(s)')
    print('    predictions:          ', '{:.3f}'.format(t4-t3), '(s)')


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        type=str, 
                        default='polymlp.in',
                        help='Input file name')
    args = parser.parse_args()

    run_sequential_generator_multiple_datasets(args.infile)


