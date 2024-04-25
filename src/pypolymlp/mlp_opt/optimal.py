#!/usr/bin/env python
import numpy as np
import argparse
import os
import yaml

from scipy.spatial import ConvexHull

def find_optimal_mlps(dirs, key, use_force=False):

    d_array = []
    for dir_pot in dirs:
        fname1 = dir_pot + '/polymlp_error.yaml'
        fname2 = dir_pot + '/polymlp_cost.yaml'
        match_d = None
        if os.path.exists(fname1) and os.path.exists(fname2):
            print(dir_pot)
            yml_data = yaml.safe_load(open(fname1))
            for d in yml_data['prediction_errors']:
                if key in d['dataset']:
                    match_d = d
                    break
            
        if match_d is not None:
            rmse_e = match_d['rmse_energy']
            rmse_f = match_d['rmse_force']
            yml_data = yaml.safe_load(open(fname2))
            time1 = yml_data['costs']['single_core']
            time36 = yml_data['costs']['openmp']
            name = dir_pot.split('/')[-1]
            d_array.append([time1, time36, rmse_e, rmse_f, name])

    d_array = sorted(d_array, key=lambda x:x[0])
    d_array = np.array(d_array)

    os.makedirs('polymlp_opts', exist_ok=True)
    np.savetxt('polymlp_opts/all.dat', d_array, fmt='%s')
    
    if use_force:
        d_target = d_array[:,[0,3]]
    else:
        d_target = d_array[:,[0,2]]

    ch = ConvexHull(d_target)
    v_convex = np.unique(ch.simplices)

    v_convex_l = []
    for v1 in v_convex:
        lower_convex = True
        time1 = float(d_array[v1][0])
        rmse_e1, rmse_f1 = float(d_array[v1][2]), float(d_array[v1][3])
        for v2 in v_convex:
            if v1 != v2: 
                time2 = float(d_array[v2][0])
                rmse_e2, rmse_f2 = float(d_array[v2][2]), float(d_array[v2][3])
                if time2 < time1 and rmse_e2 < rmse_e1 and rmse_f2 < rmse_f1:
                    lower_convex = False
                    break
        if lower_convex == True:
            v_convex_l.append(v1)
    
    d_convex = d_array[np.array(v_convex_l)]
    d_convex = d_convex[d_convex[:,2].astype(float) < 30]
    
    np.savetxt('polymlp_opts/convexhull.dat', d_convex, fmt='%s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirs', 
                        type=str, 
                        default='.', 
                        nargs='*',
                        help='Parent directory.')
    parser.add_argument('--force', 
                        action='store_true',
                        help='Use rmse (force) to get pareto frontier.')
    parser.add_argument('--key', 
                        type=str, 
                        default=None,
                        help='Test dataset name or partial string' +
                             ' used for finding optimal MLPs.')
    args = parser.parse_args()

    find_optimal_mlps(args.dirs, args.key, use_force=args.force)


