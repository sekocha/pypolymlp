#!/usr/bin/env python 
import numpy as np
import argparse
import signal

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.str_gen.prototypes_selection import (
    prototype_selection_element,
    prototype_selection_alloy,
)


def run():

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--prototypes',
                        action='store_true',
                        help='Prototype structure generation')
    parser.add_argument('-n','--n_types',
                        type=int,
                        default=None,
                        required=True,
                        help='Number of atom types (n_types = 1,2,3)')
    parser.add_argument('-c','--comp',
                        type=float,
                        nargs='*',
                        default=None,
                        help='Composition')
    parser.add_argument('--noscreen',
                        action='store_false',
                        help='All nonequivalent prototypes are generated.')

    args = parser.parse_args()

    if args.prototypes is not None:
        if args.n_types == 1:
            prototype_selection_element(screen=args.noscreen)
        else:
            target = 'alloy' # 'ionic' must be hidden
            comp = check_compositions(args.comp, args.n_types)
            print(' composition =',  comp)
            prototype_selection_alloy(args.n_types, 
                                      target=target,
                                      screen=args.noscreen, 
                                      comp=comp)


        
