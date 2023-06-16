#!/usr/bin/env python 
import numpy as np

#from mlptools.mlpgen.prediction import Pot
#from mlptools.mlpgen.error import EstimatePredictionError
#from mlptools.mlpgen.error import EstimatePredictionErrorFromPot

from scipy.linalg.lapack import get_lapack_funcs
from sklearn.linear_model import Ridge
#from sklearn.linear_model import LassoLars

from polymlp_generator.common.math_functions import rmse

class Regression:

    def __init__(self, reg_dict, params_dict):

        self.reg_dict = reg_dict
        self.params_dict = params_dict

        self.vtrain = reg_dict['train']
        self.vtest = reg_dict['test']
        self.scaler = reg_dict['scaler']

#        self.elements = data_train.elements
        self.best_alpha = None
        self.best_pred_train = None
        self.best_pred_test = None

    def ridge(self, iprint=True):

        alphas = [pow(10, a) for a in self.params_dict['reg']['alpha']]
        coefs_array = self.__ridge_fit(X=self.vtrain['x'], 
                                       y=self.vtrain['y'], 
                                       alphas=alphas)
        best_reg, best_rmse = self.__ridge_model_selection(alphas, 
                                                           coefs_array,
                                                           iprint=iprint)

        coeffs, scales = best_reg.coef_, self.scaler.scale_
        return coeffs, scales

    def __ridge_fit(self, X=None, y=None, A=None, Xy=None, alphas=[1e-3,1e-1]):

        if X is not None and y is not None:
            print('  regression: computing inner products ...')
            n_samples, n_features = X.shape
            A = np.dot(X.T, X)
            Xy = np.dot(X.T, y)
        else:
            n_features = A.shape[0]

        print('  regression: cholesky decomposition ...')
        coefs_array = np.zeros((n_features, len(alphas)))
        alpha_prev = 0.0
        for i, alpha in enumerate(alphas):
            add = alpha - alpha_prev
            A.flat[::n_features + 1] += add
            coefs_array[:,i] = self.__solve_linear_equation(A, Xy)
            alpha_prev = alpha
        A.flat[::n_features + 1] -= alpha

        return coefs_array

    def __solve_linear_equation(self, A, b):
        posv, = get_lapack_funcs(('posv',), (A, b))
        _, x, _ = posv(A, b, lower=False,
                       overwrite_a=False,
                       overwrite_b=False)
        #x = np.linalg.solve(A, b)
        #x = scipy.linalg.solve(A, b, check_finite=False, assume_a='pos')
        return x

    def __ridge_model_selection(self, alpha_array, coefs_array, iprint=True):

        pred_train_array = np.dot(self.vtrain['x'], coefs_array).T
        pred_test_array = np.dot(self.vtest['x'], coefs_array).T
        rmse_train_array = [rmse(self.vtrain['y'], p) 
                            for p in pred_train_array]
        rmse_test_array = [rmse(self.vtest['y'], p) for p in pred_test_array]

        best_reg = Ridge()
        best_reg.intercept_ = 0.0

        idx = np.argmin(rmse_test_array)
        best_rmse = rmse_test_array[idx]
        best_reg.coef_ = coefs_array[:,idx]
        self.best_alpha = alpha_array[idx]
        self.best_pred_train = pred_train_array[idx]
        self.best_pred_test = pred_test_array[idx]
 
        if iprint == True:
            print('  regression: model selection ...')
            for a, rmse1, rmse2 in zip(alpha_array, 
                                       rmse_train_array, 
                                       rmse_test_array):
                print('  - alpha =', '{:f}'.format(a), 
                      ': rmse (train, test) =', 
                      '{:f}'.format(rmse1), '{:f}'.format(rmse2))

        return best_reg, best_rmse


#        self.size_train = sum([d.get_data_size() for d in data_train.dbatches])
#        self.size_test = sum([d.get_data_size() for d in data_test.dbatches])

#    def write_pot(self,
#                  filename_pot='mlp.pkl',
#                  filename_lammps='mlp.lammps'):
#        self.pot.save_pot(file_name=filename_pot)
#        self.pot.save_pot_for_lammps(file_name=filename_lammps)
#

    """
    def lasso(self, 
              alpha_min=-5.0, 
              alpha_max=-2.0, 
              n_alpha=10, 
              iprint=True):

        best_rmse = 1e10
        for alpha in np.logspace(alpha_min, alpha_max, num=n_alpha):
            reg = LassoLars(alpha=alpha,fit_intercept=False)
            reg.fit(self.vtrain['x'], self.vtrain['y'])
            coefs = reg.coef_
            pred_train = np.dot(self.vtrain['x'], coefs)
            pred_test = np.dot(self.vtest['x'], coefs)
            rmse_train = rmse(self.vtrain['y'], pred_train)
            rmse_test = rmse(self.vtest['y'], pred_test)
            if rmse_test < best_rmse:
                best_rmse = rmse_test
                best_reg = reg
                self.best_alpha = alpha
                self.best_pred_train = pred_train
                self.best_pred_test = pred_test
            if iprint == True:
                print(' alpha =', alpha, 
                      'rmse (train, test) =', rmse_train, rmse_test)

        self.pot = Pot(reg=best_reg, 
                       scaler=self.scaler, 
                       rmse=best_rmse,
                       di=self.di, 
                       elements=self.elements)
        return self.pot
    """

    """    
    def ridge_seq(self, 
                  alpha_min=-5.0, 
                  alpha_max=-2.0, 
                  n_alpha=10, 
                  iprint=True):

        alpha_array = np.logspace(alpha_min, alpha_max, num=n_alpha)
        coefs_array = self.ridge_fit(A=self.vtrain['xtx'],
                                     Xy=self.vtrain['xty'],
                                     alpha_array=alpha_array)
        best_coefs = self.ridge_model_selection_seq(alpha_array, 
                                                    coefs_array,
                                                    iprint=iprint)
        self.pot = Pot(coefs=best_coefs, 
                       scale=self.scaler, 
                       di=self.di, 
                       elements=self.elements)
        return self.pot

    def ridge_model_selection_seq(self, alpha_array, coefs_array, iprint=True):
        
        print(' model selection ...')
        t1 = time.time()
        # computing rmse using xtx, xty and y_sq
        rmse_train_array, rmse_test_array = [], []
        for coefs in coefs_array.T:
            mse_train = self.compute_mse(self.vtrain['xtx'],
                                         self.vtrain['xty'],
                                         self.vtrain['y_sq_norm'],
                                         self.size_train,
                                         coefs)
            mse_test = self.compute_mse(self.vtest['xtx'],
                                        self.vtest['xty'],
                                        self.vtest['y_sq_norm'],
                                        self.size_test,
                                        coefs)
            rmse_train_array.append(sqrt(mse_train))
            rmse_test_array.append(sqrt(mse_test))

        idx = np.argmin(rmse_test_array)
        best_coef = coefs_array[:,idx]
        self.best_alpha = alpha_array[idx]

        if iprint == True:
            for a, rmse1, rmse2 in zip(alpha_array, 
                                       rmse_train_array, 
                                       rmse_test_array):
                print(' alpha =', a, 'rmse (train, test) =', rmse1, rmse2)

        return best_coef

    def compute_mse(self, xtx, xty, y_sq_norm, size, coefs):
        v1 = np.dot(coefs, np.dot(xtx, coefs))
        v2 = - 2 * np.dot(coefs, xty)
        return (v1 + v2 + y_sq_norm) / size
    """

#    def get_potential_model(self):
#        return self.pot

    def get_best_alpha(self):
        return self.best_alpha

    def get_predictions(self):
        return self.best_pred_train, self.best_pred_test


