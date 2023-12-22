#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.core.parser_polymlp_params import ParamsParser

from pypolymlp.mlpgen.features import Features

"""
> $(pypolymlp)/tools/compute_features.py 
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
def compute_from_polymlp_lammps(pot, poscars, return_mlp_dict=True):

    params_dict, mlp_dict = load_mlp_lammps(filename=pot)
    params_dict['include_force'] = False
    params_dict['include_stress'] = False

    structures = [Poscar(f).get_structure() for f in poscars]
    features = Features(params_dict, structures, print_memory=False)
    if return_mlp_dict:
        return features.get_x(), mlp_dict
    return features.get_x()

def compute_from_infile(infile, poscars):
    p = ParamsParser(infile, parse_vasprun_locations=False)
    params_dict = p.get_params()
    params_dict['include_force'] = False
    params_dict['include_stress'] = False

    structures = [Poscar(f).get_structure() for f in poscars]
    features = Features(params_dict, structures, print_memory=False)
    return features.get_x()

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
                        default=None,
                        help='Input parameter settings')
    parser.add_argument('--pot', 
                        type=str, 
                        default=None,
                        help='Input parameter settings (polymlp.lammps)')
    args = parser.parse_args()

    if args.infile is not None:
        p = ParamsParser(args.infile, parse_vasprun_locations=False)
        params_dict = p.get_params()
        params_dict['include_force'] = False
        params_dict['include_stress'] = False

        structures = [Poscar(f).get_structure() for f in args.poscars]
        features = Features(params_dict, structures, print_memory=False)
        x = features.get_x()
    elif args.pot is not None:
        x = compute_from_polymlp_lammps(args.pot, args.poscars, 
                                        return_mlp_dict=False)

    print(' feature size =', x.shape)
    np.save('features', x)

