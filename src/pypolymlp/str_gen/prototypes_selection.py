#!/usr/bin/env python 
import numpy as np
import argparse
import os, shutil
import itertools
import copy
from collections import defaultdict

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import swap_elements
from pypolymlp.utils.vasp_utils import write_poscar_file

def load_prototypes(n_types=1, target='alloy', screen=False):

    list_dir = '/'.join(__file__.split('/')[:-1]) + '/prototypes/'
    if screen:
        list_dir += 'list_icsd/screened/'
    else:
        list_dir += 'list_icsd/all/'
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

def prototype_selection_alloy(n_types, target='alloy', screen=False, comp=None):

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

            match_comp = True
            if comp is not None:
                n_atoms_trial = np.array(st_dict_perm['n_atoms'])
                comp_trial = n_atoms_trial / sum(n_atoms_trial)
                match_comp = np.allclose(comp, comp_trial)

            if match_comp:
                st_dict_perm['order'] = order
                st_dicts[row].append(st_dict_perm)

    output_dir = 'prototypes/'
    os.makedirs(output_dir, exist_ok=True)

    n_str = sum([len(v) for v in st_dicts.values()])
    f = open('polymlp_prototypes.yaml', 'w')
    print('n_prototype:  ', n_str, file=f)
    print('target:       ', target, file=f)
    print('screened:     ', screen, file=f)
    try:
        print('composition:  ', list(comp), file=f)
    except:
        print('composition:  ', None, file=f)
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

def check_compositions(comp, n_types):

    if comp is not None:
        if n_types != len(comp):
            raise ValueError("n_types != len(comp)")
        comp = np.array(comp)
        if abs(sum(comp) - 1.0) > 1e-8:
            comp /= sum(comp)
    return comp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    if args.n_types == 1:
        prototype_selection_element(screen=True)
    else:
        target = 'alloy' # 'ionic' must be hidden
        comp = check_compositions(args.comp, args.n_types)
        print(' composition =',  comp)
        prototype_selection_alloy(args.n_types, target=target, 
                                  screen=True, comp=comp)



