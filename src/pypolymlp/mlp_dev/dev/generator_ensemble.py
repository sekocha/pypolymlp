#!/usr/bin/env python 
import numpy as np
import argparse
import signal
import time

from pypolymlp.mlp_dev.mlpdev_core import PolymlpDevParams
from pypolymlp.mlp_dev.regression import Regression
from pypolymlp.mlp_dev.accuracy import PolymlpDevAccuracy
from pypolymlp.mlp_dev.features_attr import write_polymlp_params_yaml

from pypolymlp.mlp_dev.dev.mlpdev_data_dev import PolymlpDevEnsemble
from pypolymlp.core.io_polymlp import save_mlp_lammps

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--infile', nargs='*', type=str,  default=['polymlp.in'],
        help='Input file name'
    )
#    parser.add_argument(
#        '--no_sequential', action='store_true', 
#        help='Use normal feature calculations'
#    )
    args = parser.parse_args()

    verbose = True

    polymlp_in = PolymlpDevParams()
    polymlp_in.parse_infiles(args.infile, verbose=True)
    polymlp_in.parse_datasets()

    t1 = time.time()
    polymlp = PolymlpDevEnsemble(polymlp_in)
    polymlp.run(n_models=100, ratio_feature_samples=0.5)

    train_reg_dict_all = polymlp.train_regression_dict
    test_reg_dict_all = polymlp.test_regression_dict
    random_indices = polymlp.random_indices


    coeffs_sum = np.zeros(polymlp.n_features)
    for train_reg, test_reg, r_indices in zip(
        train_reg_dict_all, test_reg_dict_all, random_indices
    ):

        reg = Regression(
            polymlp, 
            train_regression_dict=train_reg, 
            test_regression_dict=test_reg,
        )
        reg.ridge_seq()
        mlp_dict = reg.best_model
        coeffs_sum[r_indices] += mlp_dict['coeffs'] / mlp_dict['scales']

    t2 = time.time()

    coeffs = coeffs_sum / polymlp.n_models
    scales = np.ones(polymlp.n_features)

    '''Must be extended to hybrid models'''
    save_mlp_lammps(reg.params_dict, coeffs, scales, filename='polymlp.lammps')
    t3 = time.time()

    acc = PolymlpDevAccuracy(reg, coeffs=coeffs, scales=scales)
    acc.compute_error()
    acc.write_error_yaml(filename='polymlp_error.yaml')
    t4 = time.time()

    if not acc.is_hybrid:
        write_polymlp_params_yaml(
            acc.params_dict, filename='polymlp_params.yaml'
        )
    else:
        for i, params in enumerate(acc.params_dict):
            filename = 'polymlp_params' + str(i+1) + '.yaml'
            write_polymlp_params_yaml(params, filename=filename)

    acc.write_error_yaml(filename='polymlp_error.yaml')

    if verbose:
        print('  elapsed_time:')
        print('    features:          ', '{:.3f}'.format(t2-t1), '(s)')
        print('    regression:        ', '{:.3f}'.format(t3-t2), '(s)')
        print('    error:             ', '{:.3f}'.format(t4-t3), '(s)')


