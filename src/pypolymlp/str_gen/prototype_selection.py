#!/usr/bin/env python 
import numpy as np
import os, shutil
import argparse
import itertools
import copy
from collections import defaultdict

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import swap_elements
from pypolymlp.utils.vasp_utils import write_poscar_file

def load_prototypes(n_types=1, target='alloy', screen=False):

    list_dir = '/'.join(__file__.split('/')[:-1]) + '/prototypes/'
    if screen:
        list_dir += 'list_icsd_screened/'
    else:
        list_dir += 'list_icsd/'
    if n_types == 1:
        summary = list_dir + '1-all/summary_nonequiv'
    elif n_types == 2:
        if target == 'alloy':
            summary = list_dir + '2-alloy/summary_nonequiv'
        elif target == 'ionic':
            summary = list_dir + '2-ionic/summary_nonequiv'
    elif n_types == 3:
        if target == 'alloy':
            summary = list_dir + '3-alloy/summary_nonequiv'
        elif target == 'ionic':
            summary = list_dir + '3-ionic/summary_nonequiv'

    prototypes = np.loadtxt(summary, delimiter=',', dtype=str, skiprows=1)
    return prototypes

def prototype_selection_element(screen=False):

    prototypes = load_prototypes(n_types=1, screen=screen)

    poscar_dir = '/'.join(__file__.split('/')[:-1]) + '/prototypes/poscars/'
    output_dir = 'prototypes/'
    os.makedirs(output_dir, exist_ok=True)

    f = open('polymlp_prototypes.yaml', 'w')
    print('n_prototype:  ', prototypes.shape[0], file=f)
    print('target:       ', 'None', file=f)
    print('screened:     ', screen, file=f)
    print('', file=f)
    print('prototypes:', file=f)
    for prototype in prototypes:
        poscar = poscar_dir + 'icsd-' + prototype[0]
        shutil.copy(poscar, output_dir)
        print('- icsd_collcode:  ', prototype[0], file=f)
        print('  structure_type: ', prototype[1], file=f)
        print('  space_group:    ', prototype[-1], file=f)
        print('', file=f)
    f.close()

def prototype_selection_alloy(n_types, target='alloy', screen=False):

    prototypes = load_prototypes(n_types=n_types, 
                                 target=target, 
                                 screen=screen)
    poscar_dir = '/'.join(__file__.split('/')[:-1]) + '/prototypes/poscars/'

    st_dicts = defaultdict(list)
    for row, prototype in enumerate(prototypes):
        poscar = poscar_dir + 'icsd-' + prototype[0]
        st_dict = Poscar(poscar).get_structure()
        n_atoms = np.array(st_dict['n_atoms'])

        ''' todo: equivalency of site symmetries should be examined.'''
        orders = []
        uniq = set()
        for p1 in itertools.permutations(range(n_types)): 
            cand = tuple(n_atoms[np.array(p1)])
            if not cand in uniq:
                uniq.add(cand)
                orders.append(p1)

        for order in orders:
            st_dict_perm = copy.deepcopy(st_dict)
            st_dict_perm = swap_elements(st_dict_perm, order=order)
            st_dict_perm['order'] = order
            st_dicts[row].append(st_dict_perm)

#                match_comp = True
#                if args.comp is not None:
#                    n_atoms2 = st_perm['n_atoms']
#                    comp2 = np.array(n_atoms2) / sum(n_atoms2)
#                    match_comp = np.allclose(comp2, args.comp)
#
#                if match_comp == True:
#                    order_str = ''.join([str(o) for o in order])
#                    filename2 = filename + '-' + order_str
#                    header2 = header + ' : ' + order_str
#                    print_poscar_tofile(st_perm, 
#                                        filename=filename2,
#                                        header=header2)
#

    output_dir = 'prototypes/'
    os.makedirs(output_dir, exist_ok=True)

    n_str = sum([len(v) for v in st_dicts.values()])

    f = open('polymlp_prototypes.yaml', 'w')
    print('n_prototype:  ', n_str, file=f)
    print('target:       ', target, file=f)
    print('screened:     ', screen, file=f)
    print('', file=f)
    print('prototypes:', file=f)
    for row, strs in st_dicts.items():
        prototype = prototypes[row]
        poscar_id = 'icsd-' + prototype[0]
        for st_dict in strs:
            p_str = ''.join([str(p) for p in st_dict['order']])
            filename = output_dir + poscar_id + '-' + p_str
            header = poscar_id + '-' + p_str
            write_poscar_file(st_dict, filename=filename, header=header)
        print('- icsd_collcode:  ', prototype[0], file=f)
        print('  structure_type: ', prototype[1], file=f)
        print('  space_group:    ', prototype[-1], file=f)
        print('  n_structures:   ', len(strs), file=f)
        print('', file=f)
    f.close()


if __name__ == '__main__':

    n_types = 3
    target = 'alloy' # ionic must be hidden

    if n_types == 1:
        prototype_selection_element(screen=True)
    else:
        prototype_selection_alloy(n_types, target=target, screen=True)

#
#    parser = argparse.ArgumentParser()
#    parser.add_argument('-f','--summary_file',
#                        type=str,
#                        help='Prototype list')
#    parser.add_argument('-c','--comp',
#                        type=float,
#                        nargs='*',
#                        default=None,
#                        help='Composition')
#    parser.add_argument('--max_n_atoms', 
#                        type=int, 
#                        default=64,
#                        help='Maximum number of atoms')
#
#    args = parser.parse_args()
#
#    if args.comp is not None:
#        n_type_comp = len(args.comp)
#        args.comp = np.array(args.comp)
#        if abs(sum(args.comp) - 1.0) > 1e-8:
#            args.comp /= sum(args.comp)
#        print(' composition =',  args.comp)
#
#    plist = np.loadtxt(args.summary_file, 
#                       delimiter=',', 
#                       skiprows=1, 
#                       dtype='str')
#
#    ids = plist[:,0]
#    st_types = plist[:,1]
#    poscar_dir = '/'.join(__file__.split('/')[:-1]) + '/icsd_entries/poscars/'
#    poscar_header = 'icsd-'
#    output_dir = './poscars-prototypes/'
#    os.makedirs(output_dir, exist_ok=True)
#
#    screen = []
#    for i, st_type in zip(ids, st_types):
#        poscar = poscar_dir + poscar_header + i
#        st = Poscar(poscar).get_structure()
#        if sum(st['n_atoms']) <= args.max_n_atoms:
#            screen.append((i, st_type, st))
#
#    n_st = 0
#    for i, st_type, st in screen:
#        filename = output_dir + 'poscar-' + i
#        header = poscar_header + i + ':' + st_type
#        n_types = len(st['n_atoms'])
#        if n_types == 1:
#            print_poscar_tofile(st, filename=filename, header=header)
#            n_st += 1
#        else:
#            uniq = set()
#            perm_active = []
#            # todo: equivalency of site symmetries should be examined.
#            for p1 in itertools.permutations(range(n_types)): 
#                cand = tuple(sorted(zip(st['n_atoms'], p1)))
#                if not cand in uniq:
#                    uniq.add(cand)
#                    perm_active.append(p1)
#
#            for p in perm_active:
#                st_perm = copy.deepcopy(st)
#                order = np.zeros(len(p), dtype=int)
#                for k, p1 in enumerate(p):
#                   order[p1] = k
#
#                st_perm = permute_atoms(st_perm, order)
#
#                match_comp = True
#                if args.comp is not None:
#                    n_atoms2 = st_perm['n_atoms']
#                    comp2 = np.array(n_atoms2) / sum(n_atoms2)
#                    match_comp = np.allclose(comp2, args.comp)
#
#                if match_comp == True:
#                    order_str = ''.join([str(o) for o in order])
#                    filename2 = filename + '-' + order_str
#                    header2 = header + ' : ' + order_str
#                    print_poscar_tofile(st_perm, 
#                                        filename=filename2,
#                                        header=header2)
#                    n_st += 1
#    print(' number of structures =', n_st)
#
#
