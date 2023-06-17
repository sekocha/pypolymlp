#!/usr/bin/env python
import numpy as np
import os

from polymlp_generator.common.math_functions import rmse

def __compute_rmse(true_values, 
                   pred_values_all, 
                   weight_all,
                   begin_id, 
                   end_id,
                   normalize=None,
                   return_values=False):

    pred_values = pred_values_all[begin_id:end_id]
    weight = weight_all[begin_id:end_id]

    pred = pred_values / weight
    if normalize is None:
        true = true_values
    else:
        true = true_values / np.array(normalize) 
        pred /= np.array(normalize)
    if return_values:
        return rmse(true, pred), true, pred
    return rmse(true, pred)

def print_error(error_dict, key='train'):

    print('  prediction:', key)
    print('    rmse_energy:', 
           "{0:13.7f}".format(error_dict['energy'] * 1000), '(meV/atom)')
    if error_dict['force'] is not None:
        print('    rmse_force: ', 
               "{0:13.7f}".format(error_dict['force']), '(eV/ang)')
    if error_dict['stress'] is not None:
        print('    rmse_stress:', 
               "{0:13.7f}".format(error_dict['stress'] * 1000), '(meV/atom)')

def compute_error(reg_dict, dft_dict, params_dict, mlp_dict, key='train'):

    predictions_all = mlp_dict['predictions'][key]
    weights_all = reg_dict['weight']

    n_data = len(predictions_all)
    if params_dict['include_force']:
        ebegin, fbegin, sbegin = reg_dict['first_indices'][0]
        eend, fend, send = sbegin, n_data, fbegin
    else:
        ebegin, eend = 0, n_data

    n_total_atoms = [sum(st['n_atoms'])
                     for st in dft_dict['structures']]
    rmse_e, true_e, pred_e = __compute_rmse(dft_dict['energy'],
                                            predictions_all,
                                            weights_all,
                                            ebegin, eend,
                                            normalize=n_total_atoms,
                                            return_values=True)

    rmse_f, rmse_s = None, None
    if params_dict['include_force']:
        rmse_f = __compute_rmse(dft_dict['force'],
                                predictions_all,
                                weights_all,
                                fbegin, fend)

    if params_dict['include_stress']:
        stress_unit = 'eV'
        if stress_unit == 'eV':
            normalize = np.repeat(n_total_atoms, 6)
        elif stress_unit == 'GPa':
            eV_to_GPa = 160.21766208
            volumes = [st['volume'] for st in dft_dict['structures']]
            normalize = np.repeat(volumes, 6)/eV_to_GPa
        rmse_s = __compute_rmse(dft_dict['stress'],
                                predictions_all,
                                weights_all,
                                sbegin, send,
                                normalize=normalize)

    error_dict = dict()
    error_dict['energy'] = rmse_e
    error_dict['force'] = rmse_f
    error_dict['stress'] = rmse_s
    print_error(error_dict, key=key)

    os.makedirs('predictions', exist_ok=True)
    np.savetxt('predictions/energy.' + key + '.dat', 
                np.array([true_e, pred_e, (true_e - pred_e)*1000]).T,
                header='DFT(eV/atom), MLP(eV/atom), DFT-MLP(meV/atom)', 
                fmt='%.10f')

    return error_dict

def write_error_yaml(error_dict, filename='polymlp_error.yaml'):

    f = open(filename, 'w')
    print('prediction_errors:', file=f)
    for key, dict1 in error_dict.items():
        print('- dataset:', key, file=f)
        print('  rmse_energy: ', dict1['energy'] * 1000, file=f)
        if dict1['force'] is not None:
            print('  rmse_force:  ', dict1['force'], file=f)
        if dict1['stress'] is not None:
            print('  rmse_stress: ', dict1['stress'] * 1000, file=f)
        print('', file=f)

    print('units:', file=f)
    print('  energy: meV/atom', file=f)
    print('  force:  eV/angstrom', file=f)
    print('  stress: meV/atom', file=f)
    print('', file=f)
    f.close()
    
