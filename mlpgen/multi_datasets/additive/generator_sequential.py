#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.mlpgen.parser import ParamsParser
from pypolymlp.mlpgen.multi_datasets.parser import parse_observations
from pypolymlp.mlpgen.multi_datasets.additive.parser \
                                            import set_common_params_dict
from pypolymlp.mlpgen.multi_datasets.additive.parser import print_common_params

from pypolymlp.mlpgen.multi_datasets.additive.sequential import Sequential
from pypolymlp.mlpgen.regression import Regression
from pypolymlp.mlpgen.multi_datasets.additive.io_potential \
                                            import save_multiple_mlp_lammps

from pypolymlp.mlpgen.multi_datasets.additive.accuracy \
                                            import compute_predictions
from pypolymlp.mlpgen.accuracy import compute_error
from pypolymlp.mlpgen.accuracy import write_error_yaml
from pypolymlp.mlpgen.features_attr import write_polymlp_params_yaml

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile',
                        nargs='*',
                        type=str,
                        default=['polymlp.in'],
                        help='Input file name')
    args = parser.parse_args()

    multiple_params_dicts = [ParamsParser(infile, multiple_datasets=True)
                            .get_params() for infile in args.infile]
    common_params_dict = set_common_params_dict(multiple_params_dicts)
    print_common_params(common_params_dict, infile=args.infile[0])

    train_dft_dict, test_dft_dict = parse_observations(common_params_dict)

    t1 = time.time()
    seq_train = Sequential(multiple_params_dicts, train_dft_dict)
    train_reg_dict = seq_train.get_updated_regression_dict()
    seq_test = Sequential(multiple_params_dicts, 
                          test_dft_dict,
                          scales=train_reg_dict['scales'])
    test_reg_dict = seq_test.get_updated_regression_dict()

    t2 = time.time()
    reg = Regression(train_reg_dict, test_reg_dict, common_params_dict)
    coeffs, scales = reg.ridge_seq()
    mlp_dict = reg.get_best_model()

    save_multiple_mlp_lammps(multiple_params_dicts,
                             train_reg_dict['cumulative_n_features'],
                             coeffs,
                             scales)

    print('  regression: best model')
    print('    alpha: ', mlp_dict['alpha'])

    t3 = time.time()
    error_dict = dict()
    error_dict['train'], error_dict['test'] = dict(), dict()
    for set_id, dft_dict in train_dft_dict.items():
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])\
                        .replace('..','')
        predictions, weights, indices \
                    = compute_predictions(multiple_params_dicts, 
                                          dft_dict, 
                                          coeffs, 
                                          scales)
        error_dict['train'][set_id] = compute_error(dft_dict, 
                                                    common_params_dict, 
                                                    predictions, 
                                                    weights,
                                                    indices,
                                                    output_key=output_key)
    for set_id, dft_dict in test_dft_dict.items():
        output_key = '.'.join(set_id.split('*')[0].split('/')[:-1])\
                        .replace('..','')
        predictions, weights, indices \
                        = compute_predictions(multiple_params_dicts, 
                                              dft_dict, 
                                              coeffs, 
                                              scales)
        error_dict['test'][set_id] = compute_error(dft_dict, 
                                                   common_params_dict, 
                                                   predictions, 
                                                   weights,
                                                   indices,
                                                   output_key=output_key)

    t4 = time.time()
    write_error_yaml(error_dict['train'])
    write_error_yaml(error_dict['test'], initialize=False)

    for i, params_dict in enumerate(multiple_params_dicts):
        filename = 'polymlp_params'+str(i+1)+'.yaml'
        write_polymlp_params_yaml(params_dict, filename=filename)

    print('  elapsed_time:')
    print('    features + weighting: ', '{:.3f}'.format(t2-t1), '(s)')
    print('    regression:           ', '{:.3f}'.format(t3-t2), '(s)')
    print('    predictions:          ', '{:.3f}'.format(t4-t3), '(s)')

