#!/usr/bin/env python
import numpy as np
import os, sys
import itertools
import argparse

def write_params_dict(params_dict, filename):

    f = open(filename,'w')

    print('feature_type', params_dict['feature_type'], file=f)
    print('cutoff', params_dict['cutoff'], file=f)
    
    gauss1, gauss2 = params_dict['gauss1'], params_dict['gauss2']
    print('gaussian_params1', gauss1[0], gauss1[1], gauss1[2], file=f)
    print('gaussian_params2', gauss2[0], gauss2[1], gauss2[2], file=f)

    reg = params_dict['reg_alpha_params']
    print('reg_alpha_params', reg[0], reg[1], reg[2], file=f)
    print('', file=f)

    print('model_type', params_dict['model_type'], file=f)
    print('max_p', params_dict['max_p'], file=f)
    if params_dict['feature_type'] == 'gtinv':
        print('gtinv_order', params_dict['gtinv_order'], file=f)
        print('gtinv_maxl', end='', file=f)
        for maxl in params_dict['gtinv_maxl']:
            print('', maxl, end='', file=f)
        print('', file=f)

    print('', file=f)

    print('include_force', params_dict['include_force'], file=f) 
    print('include_stress', params_dict['include_stress'], file=f) 

    f.close()

def write_grid(params_grid_pair, params_grid_gtinv):

    for i, params_dict in enumerate(params_grid_pair):
        idx = str(i + 1).zfill(4)
        dirname = 'model_grid/pair-' + idx + '/'
        os.makedirs(dirname, exist_ok=True)
        write_params_dict(params_dict, dirname + 'polymlp.in')

    for i, params_dict in enumerate(params_grid_gtinv):
        idx = str(i + 1).zfill(4)
        dirname = 'model_grid/gtinv-' + idx + '/'
        os.makedirs(dirname, exist_ok=True)
        write_params_dict(params_dict, dirname + 'polymlp.in')


def set_pair_grid(grid_setting, max_p=[2,3]):

    model_types = [t for t in grid_setting['model_types'] if t < 3]
    product = itertools.product(*[grid_setting['cutoffs'],
                                  grid_setting['n_gaussians'],
                                  model_types,
                                  max_p])

    params_grid_pair = []
    for cut, n_gauss, model_type, mp in product:
        params_dict = dict()
        params_dict['cutoff'] = cut
        params_dict['gauss2'] = (0.0, cut-1.0, n_gauss)
        params_dict['model_type'] = model_type
        params_dict['max_p'] = mp

        ''' constant settings'''
        params_dict['feature_type'] = 'pair'
        params_dict['gauss1'] = (1.0, 1.0, 1)
        params_dict['reg_alpha_params'] = grid_setting['reg_alpha_params']
        params_dict['include_force'] = grid_setting['include_force']
        params_dict['include_stress'] = grid_setting['include_stress']
        params_grid_pair.append(params_dict)

    return params_grid_pair

def set_gtinv_grid(grid_setting):

    product = itertools.product(*[grid_setting['cutoffs'],
                                  grid_setting['n_gaussians'],
                                  grid_setting['gtinv_grid']])

    params_grid_gtinv = []
    for cut, n_gauss, gtinv_dict in product:
        params_dict = dict()
        params_dict['cutoff'] = cut
        params_dict['gauss2'] = (0.0, cut-1.0, n_gauss)

        ''' constant settings'''
        params_dict['feature_type'] = 'gtinv'
        params_dict['max_p'] = 2
        params_dict['gauss1'] = (1.0, 1.0, 1)
        params_dict['reg_alpha_params'] = grid_setting['reg_alpha_params']
        params_dict['include_force'] = grid_setting['include_force']
        params_dict['include_stress'] = grid_setting['include_stress']
        params_dict.update(gtinv_dict)
        params_grid_gtinv.append(params_dict)

    return params_grid_gtinv

def set_gtinv_params(grid_setting):

    model_types = grid_setting['model_types'] 
    max_gtinv_order = grid_setting['gtinv_order_ub']
    maxl = grid_setting['gtinv_l_ub']
    maxl_intervals = grid_setting['gtinv_l_int']

    for l1, l2 in zip(maxl, maxl_intervals):
        if l1 < l2:
            raise ValueError('maxl < maxl_intervals')

    l_cand = []
    for gtinv_order in range(2, max_gtinv_order + 1):
        idx = gtinv_order - 2
        interval, ub = maxl_intervals[idx], maxl[idx]
        l_cand.append(list(range(interval, ub + 1, interval)))

    l_all = []
    for gtinv_order in range(2, max_gtinv_order + 1):
        end_idx = gtinv_order - 1
        l_prods = np.array(list(itertools.product(*l_cand[:end_idx])))
        l_prods = np.array([lp for lp in l_prods
                            if np.all(lp[:-1] - lp[1:] >= 0)])
        l_all.extend(l_prods)

    gtinv_grid = []
    for model, lp in itertools.product(model_types, l_all):
        include = True
        if model == 2 and len(lp) > 2:
            include = False
        elif model == 2 and len(lp) == 1:
            include = False

        if include:
            gtinv_dict = dict()
            gtinv_dict['model_type'] = model
            gtinv_dict['gtinv_order'] = len(lp) + 1
            gtinv_dict['gtinv_maxl'] = lp
            gtinv_grid.append(gtinv_dict)
        
    return gtinv_grid


if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('--stress', 
                    action='store_true',
                    help='include_stress True')
    ps.add_argument('--cutoff_linspace',
                    nargs=3,
                    type=float,
                    default=[6.0,10.0,3],
                    help='Cutoff parameters')
    ps.add_argument('--reg_alpha_params',
                    nargs=3,
                    type=float,
                    default=[-3,1,5],
                    help='Regularization parameters')
    ps.add_argument('--n_gauss', 
                    nargs='*', 
                    type=int, 
                    default=[7,10,13],
                    help='Number of Gaussians')
    ps.add_argument('--model_types',
                    nargs='*',
                    type=int, 
                    default=[2,3,4],
                    help='Model type parameters')
    ps.add_argument('--max_gtinv_order',
                    type=int, 
                    default=4,
                    help='Maximum order of invariants')
    ps.add_argument('--max_gtinv_maxl',
                    nargs='*',
                    type=int, 
                    default=[12,8,2,1,1],
                    help='Maximum l values of invariants')
    ps.add_argument('--interval_gtinv_maxl',
                    type=int, 
                    nargs='*',
                    default=[4,4,2,1,1],
                    help='Interval l values of invariants')
    args = ps.parse_args()

    n_gaussians = args.n_gauss
    n_gaussians[2] = int(n_gaussians[2])
    reg_alpha_params = args.reg_alpha_params 
    reg_alpha_params[2] = int(reg_alpha_params[2])

    grid_setting = dict()
    grid_setting['cutoffs'] = np.linspace(args.cutoff_linspace[0], 
                                          args.cutoff_linspace[1], 
                                          int(args.cutoff_linspace[2]))
    grid_setting['n_gaussians'] = n_gaussians
    grid_setting['reg_alpha_params'] = reg_alpha_params
    grid_setting['model_types'] = args.model_types
    grid_setting['include_force'] = True
    grid_setting['include_stress'] = args.stress

    grid_setting['gtinv_order_ub'] = args.max_gtinv_order
    grid_setting['gtinv_l_ub'] = args.max_gtinv_maxl
    grid_setting['gtinv_l_int'] = args.interval_gtinv_maxl

    params_grid_pair = set_pair_grid(grid_setting)
    if grid_setting['gtinv_order_ub'] > 1:
        grid_setting['gtinv_grid'] = set_gtinv_params(grid_setting)
        params_grid_gtinv = set_gtinv_grid(grid_setting)

    write_grid(params_grid_pair, params_grid_gtinv)


