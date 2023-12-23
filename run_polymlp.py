#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.mlp_gen.multi_datasets.generator import (
        run_generator_multiple_datasets
)
from pypolymlp.mlp_gen.multi_datasets.generator_sequential import (
        run_sequential_generator_multiple_datasets
)
from pypolymlp.mlp_gen.multi_datasets.additive.generator import (
        run_generator_additive
)
from pypolymlp.mlp_gen.multi_datasets.additive.generator_sequential import (
        run_sequential_generator_additive
)

from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.interface_vasp import parse_structures_from_vaspruns
from pypolymlp.calculator.compute_features import (
        compute_from_infile,
        compute_from_polymlp_lammps,
)
from pypolymlp.calculator.compute_properties import convert_stresses_in_gpa
from pypolymlp.calculator.compute_properties import compute_properties
from pypolymlp.calculator.compute_fcs import compute_fcs


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', 
                        nargs='*',
                        type=str, 
                        default=['polymlp.in'],
                        help='Input file name')
    parser.add_argument('--sequential', 
                        action='store_true',
                        help='Use sequential evaluation of X.T @ X')
    parser.add_argument('--features', 
                        action='store_true',
                        help='Mode: Feature calculation')
    parser.add_argument('--properties', 
                        action='store_true',
                        help='Mode: Property calculation')
    parser.add_argument('--force_constants', 
                        action='store_true',
                        help='Mode: Force constant calculation')

    parser.add_argument('--pot',
                        type=str,
                        default='polymlp.lammps',
                        help='polymlp file')
    parser.add_argument('--poscars',
                        nargs='*',
                        type=str,
                        default=None,
                        help='poscar files')
    parser.add_argument('--vaspruns',
                        nargs='*',
                        type=str,
                        default=None,
                        help='vasprun files')
    parser.add_argument('--phono3py_yaml',
                        type=str,
                        default=None,
                        help='phono3py.yaml file')
    args = parser.parse_args()

    if (args.features == False and args.properties == False 
        and args.force_constants == False):
        if len(args.infile) == 1:
            infile = args.infile[0]
            if args.sequential:
                print('Mode: Sequential regression')
                run_sequential_generator_multiple_datasets(infile)
            else:
                print('Mode: Regression')
                run_generator_multiple_datasets(infile)
        else:
            if args.sequential:
                print('Mode: Sequential regression (additive model)')
                run_sequential_generator_additive(args.infile)
            else:
                print('Mode: Regression (additive model)')
                run_generator_additive(args.infile)

    if args.properties:
        print('Mode: Property calculations')
        if args.poscars is not None:
            structures = parse_structures_from_poscars(args.poscars)
        elif args.vaspruns is not None:
            structures = parse_structures_from_vaspruns(args.vaspruns)
        elif args.phono3py_yaml is not None:
            from pypolymlp.core.interface_phono3py import (
                parse_structures_from_phono3py_yaml
            )
            structures = parse_structures_from_phono3py_yaml(args.phono3py_yaml)

        energies, forces, stresses = compute_properties(args.pot, structures)
        stresses_gpa = convert_stresses_in_gpa(stresses, structures)
        np.set_printoptions(suppress=True)
        ''' todo: output format should be considered. '''
        print(energies)

    if args.force_constants:
        print('Mode: Force constant calculations')
        compute_fcs(args.pot, phono3py_yaml=args.phono3py_yaml)

    if args.features:
        print('Mode: Feature matrix calculations')
        structures = parse_structures_from_poscars(args.poscars)
        if args.pot is not None:
            x = compute_from_polymlp_lammps(args.pot, structures,
                                            return_mlp_dict=False)
        else:
            infile = args.infile[0]
            x = compute_from_infile(infile, structures)

        print(' feature size =', x.shape)
        np.save('features.npy', x)

    ''' todo: args.strgen should be implemented.'''


