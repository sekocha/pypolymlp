#!/usr/bin/env python
import numpy as np
import os
import shutil
import yaml
import itertools
import argparse

from pypolymlp.mlp_opt.grid_io import write_params_dict
from pypolymlp.mlp_opt.grid_search_single import set_gtinv_params


def set_gtinv_grid_hybrid(grid_setting):

    product = itertools.product(*[grid_setting['cutoffs'],
                                  grid_setting['n_gaussians'],
                                  grid_setting['gtinv_grid']])

    params_grid_gtinv = []
    for cut, n_gauss, gtinv_dict in product:
        params_dict = dict()
        params_dict['cutoff'] = cut

        if n_gauss == 1:
            min1 = max1 = cut/2
        else:
            min1 = 1.0
            max1 = cut - 1.0
        params_dict['gauss2'] = (min1, max1, n_gauss)

        ''' constant settings'''
        params_dict['feature_type'] = 'gtinv'
        params_dict['max_p'] = 2
        params_dict['gauss1'] = (0.5, 0.5, 1)
        params_dict['reg_alpha_params'] = grid_setting['reg_alpha_params']
        params_dict['include_force'] = grid_setting['include_force']
        params_dict['include_stress'] = grid_setting['include_stress']
        params_dict.update(gtinv_dict)
        params_grid_gtinv.append(params_dict)

    return params_grid_gtinv



if __name__ == '__main__':

    ps = argparse.ArgumentParser()
    ps.add_argument('--yaml',
                    type=str,
                    default='polymlp_summary_convex.yaml',
                    help='Summary yaml file')
    ps.add_argument('--no_stress',
                    action='store_false',
                    help='Stress')
    args = ps.parse_args()

    f = open(args.yaml)
    yamldata = yaml.safe_load(f)
    f.close()

    grid_setting_h = dict()
    grid_setting_h['cutoffs'] = [3.0,4.0,5.0]
    grid_setting_h['n_gaussians'] = [2,4]
    grid_setting_h['reg_alpha_params'] = [-4.0,3.0,15]
    grid_setting_h['model_types'] = [4]
    grid_setting_h['include_force'] = True
    grid_setting_h['include_stress'] = args.no_stress

    grid_setting_h['gtinv_order_ub'] = 3
    grid_setting_h['gtinv_l_ub'] = [8,8,2,1,1]
    grid_setting_h['gtinv_l_int'] = [8,8,2,1,1]
    grid_setting_h['gtinv_grid'] = set_gtinv_params(grid_setting_h)

    params_grid_hybrid = set_gtinv_grid_hybrid(grid_setting_h)

    polymlps = yamldata['polymlps']
    for pot in polymlps:
        f = open(pot['path'] + '/polymlp.in')
        lines = f.readlines()
        f.close()

        addlines = []
        for l in lines:
            if 'n_type' in l:
                addlines.append(l)
            if 'elements' in l:
                addlines.append(l)

        for i, params in enumerate(params_grid_hybrid):
            path_output = pot['id'] + '-hybrid-' + str(i+1).zfill(4)
            os.makedirs(path_output, exist_ok=True)
            shutil.copy(pot['path'] + '/polymlp.in', path_output)
            write_params_dict(params, path_output + '/polymlp.in.2')
            f = open(path_output + '/polymlp.in.2', 'a')
            print('', file=f)
            for l in addlines:
                print(l, file=f, end='')
            print('', file=f)
            f.close()


