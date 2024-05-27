#!/usr/bin/env python 
import numpy as np
import time

from pypolymlp.mlp_dev.learning_curve import learning_curve


if __name__ == '__main__':

    import argparse
    import signal

    from pypolymlp.mlp_dev.mlpdev_core import PolymlpDevParams
    from pypolymlp.mlp_dev.mlpdev_data import PolymlpDev
    from pypolymlp.mlp_dev.regression import Regression
    from pypolymlp.mlp_dev.accuracy import PolymlpDevAccuracy
    from pypolymlp.mlp_dev.features_attr import write_polymlp_params_yaml


    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        nargs='*',
                        type=str, 
                        default=['polymlp.in'],
                        help='Input file name')
    args = parser.parse_args()

    verbose = True
    no_sequential = True

    polymlp_in = PolymlpDevParams()
    polymlp_in.parse_infiles(args.infile, verbose=True)

    '''must be revised'''
    polymlp_in.parse_datasets()

    t1 = time.time()
    if no_sequential == True:
        polymlp = PolymlpDev(polymlp_in).run()
        polymlp.print_data_shape()
    else:
        pass
    t2 = time.time()

    reg = Regression(polymlp).ridge()
    reg.save_mlp_lammps(filename='polymlp.lammps')
    t3 = time.time()

    if verbose:
        mlp_dict = reg.best_model
        print('  Regression: best model')
        print('    alpha: ', mlp_dict['alpha'])

    acc = PolymlpDevAccuracy(reg)
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


