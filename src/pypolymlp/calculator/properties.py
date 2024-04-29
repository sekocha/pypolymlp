#!/usr/bin/env python 
import numpy as np

from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.calculator.compute_features import update_types
from pypolymlp.cxx.lib import libmlpcpp


def convert_stresses_in_gpa(stresses, st_dicts):

    volumes = np.array([st['volume'] for st in st_dicts])
    stresses_gpa = np.zeros(stresses.shape)
    for i in range(6):
        stresses_gpa[:,i] = stresses[:,i] / volumes * 160.21766208
    return stresses_gpa


class PropertiesSingle:

    def __init__(self, pot=None, params_dict=None, coeffs=None):

        if pot is not None:
            self.__params_dict, mlp_dict = load_mlp_lammps(filename=pot)
            self.__coeffs = mlp_dict['coeffs'] / mlp_dict['scales']
        else:
            self.__params_dict = params_dict
            self.__coeffs = coeffs

        self.__params_dict['element_swap'] = False
        self.obj = libmlpcpp.PotentialPropertiesFast(self.__params_dict,
                                                     self.__coeffs)

    def eval(self, st_dict):
        '''
        Return
        ------
        energy: unit: eV/supercell 
        force: unit: eV/angstrom (3, n_atom)
        stress: unit: eV/supercell: (6) in the order of xx, yy, zz, xy, yz, zx
        '''
        element_order = self.__params_dict['elements']
        st_dict = update_types([st_dict], element_order)[0]

        positions_c = st_dict['axis'] @ st_dict['positions']
        self.obj.eval(st_dict['axis'], positions_c, st_dict['types'])

        energy = self.obj.get_e()
        force = np.array(self.obj.get_f()).T
        stress = np.array(self.obj.get_s())
        return energy, force, stress

    def eval_multiple(self, st_dicts):
        '''
        Return
        ------
        energies: unit: eV/supercell (n_str)
        forces: unit: eV/angstrom (n_str, 3, n_atom)
        stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                    unit: eV/supercell
        '''
        print('Properties calculations for', len(st_dicts), 
              'structures: Using a fast algorithm')
        element_order = self.__params_dict['elements']
        st_dicts = update_types(st_dicts, element_order)

        axis_array = [st['axis'] for st in st_dicts]
        types_array = [st['types'] for st in st_dicts]
        positions_c_array = [st['axis'] @ st['positions'] for st in st_dicts]

        '''    
        PotentialProperties.eval_multiple: Return
        ------------------------------------------
        energies = obj.get_e(), (n_str)
        forces = obj.get_f(), (n_str, n_atom, 3)
        stresses = obj.get_s(), (n_str, 6) 
                    in the order of xx, yy, zz, xy, yz, zx
        '''
        self.obj.eval_multiple(axis_array, positions_c_array, types_array)

        energies = np.array(self.obj.get_e_array())
        stresses = np.array(self.obj.get_s_array())
        forces = [np.array(f).T for f in self.obj.get_f_array()]
        return energies, forces, stresses

    @property
    def params_dict(self):
        return self.__params_dict


class PropertiesHybrid:

    def __init__(self, pot=None, params_dict=None, coeffs=None):

        if pot is not None:
            try:
                bool_list = isinstance(pot, list)
            except:
                raise ValueError(
                    'Parameters in PropertiesHybrid must be lists.'
                )
            self.props = [PropertiesSingle(pot=p) for p in pot]
        else:
            try:
                bool_list = isinstance(params_dict, list)
                bool_list = isinstance(coeffs, list)
            except:
                raise ValueError(
                    'Parameters in PropertiesHybrid must be lists.'
                )

            self.props = [PropertiesSingle(params_dict=p, coeffs=c) 
                           for p, c in zip(params_dict, coeffs)]

    def eval(self, st_dict):

        energy, force, stress = self.props[0].eval(st_dict)
        for prop in self.props[1:]:
            e_single, f_single, s_single = prop.eval(st_dict)
            energy += e_single
            force += f_single
            stress += s_single

        return energy, force, stress


    def eval_multiple(self, st_dicts):
 
        energies, forces, stresses = self.props[0].eval_multiple(st_dicts)
        for prop in self.props[1:]:
            e_single, f_single, s_single = prop.eval_multiple(st_dicts)
            energies += e_single
            for i, f1 in enumerate(f_single):
                forces[i] += f1
            stresses += s_single

        return energies, forces, stresses

    @property
    def params_dict(self):
        return [prop.params_dict for prop in self.props]


class Properties:

    def __init__(self, pot=None, params_dict=None, coeffs=None):
        
        if pot is not None:
            if isinstance(pot, list): 
                if len(pot) > 1:
                    self.prop = PropertiesHybrid(pot=pot)
                else:
                    self.prop = PropertiesSingle(pot=pot[0])
            else:
                self.prop = PropertiesSingle(pot=pot)
        else:
            if isinstance(params_dict, list) and isinstance(coeffs, list):
                if len(params_dict) > 1 and len(coeffs) > 1:
                    self.prop = PropertiesHybrid(params_dict=params_dict,
                                                 coeffs=coeffs)
                else:
                    self.prop = PropertiesSingle(params_dict=params_dict[0],
                                                 coeffs=coeffs[0])
            else:
                self.prop = PropertiesSingle(params_dict=params_dict,
                                             coeffs=coeffs)

    def eval(self, st_dict):
        return self.prop.eval(st_dict)

    def eval_multiple(self, st_dicts):
        return self.prop.eval_multiple(st_dicts)

    def eval_phonopy(self, str_ph):
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
        st_dict = phonopy_cell_to_st_dict(str_ph)
        return self.prop.eval(st_dict)

    def eval_multiple_phonopy(self, str_ph_list):
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_st_dict
        st_dicts = [phonopy_cell_to_st_dict(str_ph) for str_ph in str_ph_list]
        return self.prop.eval_multiple(st_dicts)

    @property
    def params_dict(self):
        return self.prop.params_dict


if __name__ == '__main__':

    import argparse
    from pypolymlp.core.interface_vasp import parse_structures_from_poscars
    from pypolymlp.core.interface_vasp import parse_structures_from_vaspruns

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
                        nargs='*',
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

    prop = Properties(pot=args.pot)
    energies, forces, stresses = prop.eval_multiple(structures)
    stresses_gpa = convert_stresses_in_gpa(stresses, structures)

    np.set_printoptions(suppress=True)
    np.save('polymlp_energies.npy', energies)
    np.save('polymlp_forces.npy', forces)
    np.save('polymlp_stress_tensors.npy', stresses_gpa)

    if len(forces) == 1:
        np.savetxt('polymlp_energies.dat', energies, fmt='%f')
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


