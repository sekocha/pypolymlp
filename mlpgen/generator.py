#!/usr/bin/env python 
import numpy as np
import argparse

import signal

from polymlp_generator.common.vasp import parse_vaspruns
from polymlp_generator.mlpgen.params_parser import ParamsParser
from polymlp_generator.mlpgen.features import Features
from polymlp_generator.mlpgen.precondition import Precondition

#from mlptools.common.mathfunction import rmse
#from mlptools.mlpgen.io import ReadFeatureParams
#from mlptools.mlpgen.io import read_regression_params
#
#from mlptools.mlpgen.data import DataInput
#from mlptools.mlpgen.data_setting import DataRegression
#from mlptools.mlpgen.prediction import Pot
#from mlptools.mlpgen.error import EstimatePredictionError
#from mlptools.mlpgen.error import EstimatePredictionErrorFromPot
#
#import scipy
#from scipy.linalg.lapack import get_lapack_funcs
#from sklearn.linear_model import LinearRegression

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
    print(dft_dict['test']['energy'])

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
    scaler = pre_train.get_scaler()

    pre_test = Precondition(reg_dict['test'], 
                            dft_dict['test'], 
                            params_dict, 
                            scaler=scaler)
    pre_test.print_data_shape(header='test data size')

    print(reg_dict['train']['y'].shape)
    print(dft_dict['test']['energy'])




#    if args.sequential == False:
#        t2 = time.time()
#        data_train.compute_features()
#        data_test.compute_features()
#        t3 = time.time()
#        data_train.define_xy()
#        scaler = data_train.get_scaler(target='energy')
#        scaler_c = data_train.get_scaler(target='charge')
#        data_test.define_xy(scaler=scaler, scaler_charge=scaler_c)
#        t4 = time.time()


#    t4 = time.time()
#    reg_method, alpha_min, alpha_max, n_alpha = read_regression_params(p)
#    print(' regression for potential energy model')
#    reg_e = RegressionModel(data_train, data_test, target='energy')
#    if reg_method == 'ridge':
#        if args.sequential == False:
#            pot_e = reg_e.ridge(alpha_min=alpha_min,
#                                alpha_max=alpha_max,
#                                n_alpha=n_alpha)
#        else: 
#            pot_e = reg_e.ridge_seq(alpha_min=alpha_min,
#                                    alpha_max=alpha_max,
#                                    n_alpha=n_alpha)
#        alpha_e = reg_e.get_best_parameters()
#        pred_train, pred_test = reg_e.get_predictions()
#        print(' --- best model ----')
#        print(' alpha = ', alpha_e)
#    elif reg_method == 'normal':
#        if args.sequential == False:
#            pot_e = reg_e.normal_reg()
#        else:
#            raise KeyError('Only ridge regression is available',
#                           'for sequential regression.')
#    elif reg_method == 'huber':
#        if args.sequential == False:
#            pot_e = reg_e.huber()
#        alpha_e = reg_e.get_best_parameters()
#        pred_train, pred_test = reg_e.get_predictions()
#        print(' --- best model ----')
#        print(' alpha = ', alpha_e)
#
#    print(' --- input parameters ----')
#    pot_e.di.model_e.print()
#
#    if di.charge_model == True:
#        print(' regression for charge model')
#        reg_c = RegressionModel(data_train, data_test, target='charge')
#        if reg_method == 'ridge':
#            pot_c = reg_c.ridge(alpha_min=alpha_min,
#                                alpha_max=alpha_max,
#                                n_alpha=n_alpha)
#            pred_train_c, pred_test_c = reg_c.get_predictions()
#        elif reg_method == 'normal':
#            pot_c = reg_c.normal_reg()
#
#        #print(' --- input parameters ----')
#        #pot_c.di.model_e.print()
#
#    t5 = time.time()
#    if di.charge_model == False:
#        if args.sequential == False:
#            error = EstimatePredictionError(data_train, 
#                                            data_test, 
#                                            pred_train, 
#                                            pred_test)
#        else: 
#            error = EstimatePredictionErrorFromPot(data_train, 
#                                                   data_test,
#                                                   pot_e)
#    else:
#        error = EstimatePredictionError(data_train, 
#                                        data_test, 
#                                        pred_train, 
#                                        pred_test, 
#                                        pred_train_c, 
#                                        pred_test_c)
#
#    t6 = time.time()
#    reg_e.write_pot(filename_pot=args.pot, filename_lammps=args.lammps)
#    if di.charge_model == True:
#        reg_c.write_pot(filename_pot=args.pot+'.charge',
#                        filename_lammps=args.lammps+'.charge')
#    error.print_energy_predictions()
#    error.print_errors()
#    t7 = time.time()
#
#    print(' elapsed time (electrostatic)   =', '{:.3f}'.format(t2-t1), '(s)')
#    print(' elapsed time (features)        =', '{:.3f}'.format(t3-t2), '(s)')
#    print(' elapsed time (scaling, weight) =', '{:.3f}'.format(t4-t3), '(s)')
#    print(' elapsed time (regression)      =', '{:.3f}'.format(t5-t4), '(s)')
#    print(' elapsed time (prediction)      =', '{:.3f}'.format(t6-t5), '(s)')
#    print(' elapsed time (print files)     =', '{:.3f}'.format(t7-t6), '(s)')
#
