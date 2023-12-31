#!/usr/bin/env python 
import numpy as np
import argparse

from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.calculator.compute_features import (
        update_types,
        compute_from_polymlp_lammps,
)

from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.interface_vasp import parse_structures_from_vaspruns

from pypolymlp.mlp_gen.features import structures_to_mlpcpp_obj
from pypolymlp.cxx.lib import libmlpcpp

def compute_properties(pot, st_dicts):
    '''
    Return
    ------
    energies: unit: eV/supercell (n_str)
    forces: unit: eV/angstrom (n_str, 3, n_atom)
    stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                unit: eV/supercell
    '''
    print('Properties: Using a memory efficient algorithm')

    params_dict, mlp_dict = load_mlp_lammps(filename=pot)
    params_dict['element_swap'] = False
    coeffs = mlp_dict['coeffs'] / mlp_dict['scales']

    element_order = params_dict['elements']
    st_dicts = update_types(st_dicts, element_order)
    axis_array, positions_c_array, types_array, _ \
                        = structures_to_mlpcpp_obj(st_dicts)

    obj = libmlpcpp.PotentialProperties(params_dict,
                                        coeffs,
                                        axis_array,
                                        positions_c_array,
                                        types_array)
    '''
    PotentialProperties: Return
    ----------------------------
    energies = obj.get_e(), (n_str)
    forces = obj.get_f(), (n_str, n_atom*3) in the order of (n_atom, 3)
    stresses = obj.get_s(), (n_str, 6) 
                             in the order of xx, yy, zz, xy, yz, zx
    '''
    energies, forces, stresses = obj.get_e(), obj.get_f(), obj.get_s()
    forces = [np.array(f).reshape((-1,3)).T for f in forces]

    return energies, forces, stresses
 
def compute_energies(pot, st_dicts):

    x, mlp_dict = compute_from_polymlp_lammps(pot, st_dicts, 
                                              force=False,
                                              stress=False)
    coeffs = mlp_dict['coeffs'] / mlp_dict['scales']
    return x @ coeffs

def compute_properties_slow(pot, st_dicts):
    '''
    Return
    ------
    energies: unit: eV/supercell (n_str)
    forces: unit: eV/angstrom (n_str, 3, n_atom)
    stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                unit: eV/supercell
    '''
    features, mlp_dict = compute_from_polymlp_lammps(pot, st_dicts, 
                                                     force=True,
                                                     stress=True,
                                                     return_features_obj=True,
                                                     return_mlp_dict=True)
    x = features.get_x()
    coeffs = mlp_dict['coeffs'] / mlp_dict['scales']
    predictions = x @ coeffs

    e_ptr, f_ptr, s_ptr = features.get_first_indices()[0]
    energies = predictions[e_ptr:s_ptr]
    stresses = predictions[s_ptr:f_ptr].reshape((-1,6))

    forces = []
    begin_ptr = f_ptr
    for n_atom in features.get_n_atoms_sums():
        end_ptr = begin_ptr + n_atom * 3
        forces.append(predictions[begin_ptr:end_ptr].reshape((-1,3)).T)
        begin_ptr = end_ptr

    return (energies, forces, stresses)

def convert_stresses_in_gpa(stresses, st_dicts):

    volumes = np.array([st['volume'] for st in st_dicts])
    stresses_gpa = np.zeros(stresses.shape)
    for i in range(6):
        stresses_gpa[:,i] = stresses[:,i] / volumes * 160.21766208
    return stresses_gpa
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscars', 
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
    parser.add_argument('--pot', 
                        type=str, 
                        default='polymlp.lammps',
                        help='polymlp file')
    args = parser.parse_args()

    if args.poscars is not None:
        structures = parse_structures_from_poscars(args.poscars)
    elif args.vaspruns is not None:
        structures = parse_structures_from_vaspruns(args.vaspruns)
    elif args.phono3py_yaml is not None:
        from pypolymlp.core.interface_phono3py import (
            parse_structures_from_phono3py_yaml
        )
        structures = parse_structures_from_phono3py_yaml(args.phono3py_yaml)

    '''
    energies = compute_energies(args.pot, structures)
    '''
    energies, forces, stresses = compute_properties(args.pot, structures)
    stresses_gpa = convert_stresses_in_gpa(stresses, structures)

    np.set_printoptions(suppress=True)
    np.save('polymlp_energies.npy', energies)
    np.save('polymlp_forces.npy', forces)
    np.save('polymlp_stress_tensors.npy', stresses_gpa)

    if len(forces) == 1:
        print(' energy =', energies[0], '(eV/cell)')
        print(' forces =')
        for i, f in enumerate(forces[0].T):
            print('  - atom', i, ":", f)
        stress = stresses_gpa[0]
        print(' stress tensors =')
        print('  - xx, yy, zz:', stress[0:3])
        print('  - xy, yz, zx:', stress[3:6])
        print('---------')
        print(' polymlp_energies.npy, polymlp_forces.npy,',
              'and polymlp_stress_tensors.npy are generated.')
    else:
        print(' polymlp_energies.npy, polymlp_forces.npy,',
              'and polymlp_stress_tensors.npy are generated.')


