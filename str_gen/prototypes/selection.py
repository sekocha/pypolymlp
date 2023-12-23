#!/usr/bin/env python
import numpy as np
import os
import argparse
import itertools
import copy

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.str_gen.structure import permute_atoms
from pypolymlp.str_gen.structure import print_poscar_tofile

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--summary_file',
                        type=str,
                        help='Prototype list')
    parser.add_argument('-c','--comp',
                        type=float,
                        nargs='*',
                        default=None,
                        help='Composition')
    parser.add_argument('--max_n_atoms', 
                        type=int, 
                        default=64,
                        help='Maximum number of atoms')

    args = parser.parse_args()

    if args.comp is not None:
        n_type_comp = len(args.comp)
        args.comp = np.array(args.comp)
        if abs(sum(args.comp) - 1.0) > 1e-8:
            args.comp /= sum(args.comp)
        print(' composition =',  args.comp)

    plist = np.loadtxt(args.summary_file, 
                       delimiter=',', 
                       skiprows=1, 
                       dtype='str')

    ids = plist[:,0]
    st_types = plist[:,1]
    poscar_dir = '/'.join(__file__.split('/')[:-1]) + '/icsd_entries/poscars/'
    poscar_header = 'icsd-'
    output_dir = './poscars-prototypes/'
    os.makedirs(output_dir, exist_ok=True)

    screen = []
    for i, st_type in zip(ids, st_types):
        poscar = poscar_dir + poscar_header + i
        st = Poscar(poscar).get_structure()
        if sum(st['n_atoms']) <= args.max_n_atoms:
            screen.append((i, st_type, st))

    n_st = 0
    for i, st_type, st in screen:
        filename = output_dir + 'poscar-' + i
        header = poscar_header + i + ':' + st_type
        n_types = len(st['n_atoms'])
        if n_types == 1:
            print_poscar_tofile(st, filename=filename, header=header)
            n_st += 1
        else:
            uniq = set()
            perm_active = []
            # todo: equivalency of site symmetries should be examined.
            for p1 in itertools.permutations(range(n_types)): 
                cand = tuple(sorted(zip(st['n_atoms'], p1)))
                if not cand in uniq:
                    uniq.add(cand)
                    perm_active.append(p1)

            for p in perm_active:
                st_perm = copy.deepcopy(st)
                order = np.zeros(len(p), dtype=int)
                for k, p1 in enumerate(p):
                   order[p1] = k

                st_perm = permute_atoms(st_perm, order)

                match_comp = True
                if args.comp is not None:
                    n_atoms2 = st_perm['n_atoms']
                    comp2 = np.array(n_atoms2) / sum(n_atoms2)
                    match_comp = np.allclose(comp2, args.comp)

                if match_comp == True:
                    order_str = ''.join([str(o) for o in order])
                    filename2 = filename + '-' + order_str
                    header2 = header + ' : ' + order_str
                    print_poscar_tofile(st_perm, 
                                        filename=filename2,
                                        header=header2)
                    n_st += 1
    print(' number of structures =', n_st)



