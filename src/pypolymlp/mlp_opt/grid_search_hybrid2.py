#!/usr/bin/env python
import numpy as np
import os
import shutil
import yaml
import itertools
import argparse

from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.mlp_opt.grid_io import write_params_dict
from pypolymlp.mlp_opt.grid_search_hybrid import set_models

def model_for_pair(stress, reg_alpha_params, params_dict_in):

    cutoff = params_dict_in['model']['cutoff']
    cutoffs = [cutoff-1.0, cutoff-2.0]

    maxl_cands = [[4],[8],[4,4]]
    params_dict_all = []
    for ml, mp in itertools.product(maxl_cands, [2,3]):
        grid1 = set_models(
            cutoffs, stress, reg_alpha_params,
            n_gauss2=3, model_type=4, max_p=mp, gtinv_maxl=ml,
        )
        params_dict_all.extend(grid1)

    return params_dict_all


def model_for_gtinv(stress, reg_alpha_params, params_dict_in):

    cutoff = params_dict_in['model']['cutoff']
    maxl = params_dict_in['model']['gtinv']['max_l']

    cutoffs = [cutoff-1.0, cutoff-2.0]

    maxl_cands = []
    if len(maxl) == 1:
        maxl_cands.append([maxl[0] + 4])
        maxl_cands.append([maxl[0] + 4, 8])
    elif len(maxl) == 2:
        maxl_cands.append([maxl[0] + 4, maxl[1] + 4])
        maxl_cands.append([maxl[0] + 4, maxl[1] + 4, 4])
    elif len(maxl) == 3:
        maxl_cands.append([maxl[0] + 4, maxl[1] + 4, maxl[2]])
        maxl_cands.append([maxl[0] + 4, maxl[1] + 4, maxl[2] + 2, 1, 1])
    else:
        maxl_cands.append([l + 4 if i < 2 else l for i, l in enumerate(maxl)])

    params_dict_all = []
    for ml, mp in itertools.product(maxl_cands, [2,3]):
        grid1 = set_models(
            cutoffs, stress, reg_alpha_params,
            n_gauss2=3, model_type=4, max_p=mp, gtinv_maxl=ml,
        )
        params_dict_all.extend(grid1)

    grid2 = set_models(
        cutoffs, stress, reg_alpha_params,
        n_gauss2=3, model_type=2, max_p=2, gtinv_maxl=maxl[:2],
    )
    params_dict_all.extend(grid2)

    return params_dict_all



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

    reg_alpha_params = [-4.0,3.0,15]
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

        params = ParamsParser(
            pot['path'] + '/polymlp.in', parse_vasprun_locations=False
        )
        params_dict = params.get_params()

        if params_dict['model']['feature_type'] == 'gtinv':
            grid = model_for_gtinv(
                args.no_stress, reg_alpha_params, params_dict)
        else:
            grid = model_for_pair(args.no_stress, reg_alpha_params, params_dict)

        for i, params in enumerate(grid):
            path_output = pot['id'] + '-hybrid-' + str(i+1).zfill(4)
            os.makedirs(path_output, exist_ok=True)
            shutil.copy(pot['path'] + '/polymlp.in', path_output)
            write_params_dict(params, path_output + '/polymlp.in.2')
            f = open(path_output + '/polymlp.in.2', 'a')
            print('', file=f)
            for l in addlines:
                print(l, file=f, end='')
            print('', file=f)
