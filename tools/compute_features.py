#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from polymlp_generator.common.vasp import Poscar
from polymlp_generator.mlpgen.file_parser import ParamsParser
from polymlp_generator.mlpgen.features import Features

"""
> $(polymlp_generator)/tools/compute_features.py 
    --infile polymlp.in --poscars poscars/poscar-000*

> cat polymlp.in

    n_type 2
    elements Mg O

    feature_type gtinv
    cutoff 8.0
    model_type 3
    max_p 2

    gtinv_order 3
    gtinv_maxl 4 4

    gaussian_params1 1.0 1.0 1
    gaussian_params2 0.0 7.0 8

"""


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscars', 
                        nargs='*',
                        type=str, 
                        default=['POSCAR'],
                        help='poscar files')
    parser.add_argument('-i', '--infile', 
                        type=str, 
                        default='polymlp.in',
                        help='Input parameter settings')
    args = parser.parse_args()

    p = ParamsParser(args.infile, parse_vasprun_locations=False)
    params_dict = p.get_params()
    params_dict['include_force'] = False
    params_dict['include_stress'] = False

    structures = [Poscar(f).get_structure() for f in args.poscars]
    features = Features(params_dict, structures, print_memory=False)
    x = features.get_x()

    print(' feature size =', x.shape)
    np.save('features', x)

