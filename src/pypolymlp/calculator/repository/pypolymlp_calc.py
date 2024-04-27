#!/usr/bin/env python 
import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym
from pypolymlp.calculator.compute_elastic import PolymlpElastic
from pypolymlp.calculator.compute_phonon import (
    PolymlpPhonon, PolymlpPhononQHA
)


class PypolymlpCalc:

    def __init__(self, 
                 pot=None,
                 params_dict=None,
                 coeffs=None,
                 properties=None):

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot,
                                   params_dict=params_dict,
                                   coeffs=coeffs)
        self.pot = pot
        self.params_dict = params_dict
        self.coeffs = coeffs

    def compute_phonon(self, 
                       unitcell_dict, 
                       supercell_auto=True,
                       supercell_matrix=None,
                       disp=0.01,
                       mesh=[10,10,10],
                       tmin=100,
                       tmax=1000,
                       tstep=100,
                       pdos=False):

        if supercell_auto:
            pass
        elif supercell_matrix is None:
            raise ValueError('compute_phonon: supercell_matrix is needed.')
        
        ph = PolymlpPhonon(unitcell_dict, 
                           supercell_matrix, 
                           properties=self.prop)
        ph.produce_force_constants(displacements=disp)
        ph.compute_properties(mesh=mesh,
                              t_min=tmin,
                              t_max=tmax,
                              t_step=tstep,
                              pdos=pdos)

    def compute_phonon_qha(self,
                           unitcell_dict, 
                           supercell_auto=True,
                           supercell_matrix=None):

        qha = PolymlpPhononQHA(unitcell_dict, 
                               supercell_matrix, 
                               properties=self.prop)


def run_single_structure(st_dict, 
                         pot=None,
                         params_dict=None,
                         coeffs=None,
                         properties=None,
                         run_qha=False):

    if properties is not None:
        prop = properties
    else:
        prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)


    print('Mode: Geometry optimization')
    minobj = MinimizeSym(st_dict, properties=prop, relax_cell=True)
    minobj.run(gtol=1e-5)
    minobj.write_poscar()
    st_dict_eq = Poscar('POSCAR_eqm').get_structure()

    print('Mode: Elastic constant')
    el = PolymlpElastic(st_dict_eq, 'POSCAR_eqm', properties=prop)
    el.run()
    el.write_elastic_constants()

    print('Mode: Phonon')
    supercell_matrix = np.diag([4,4,4])
    ph = PolymlpPhonon(st_dict_eq, supercell_matrix, properties=prop)
    ph.produce_force_constants(displacements=0.01)
    ph.compute_properties(mesh=[10,10,10], t_min=100, t_max=1000, t_step=50)

    if run_qha and not ph.is_imaginary():
        print('Mode: Phonon QHA')
        qha = PolymlpPhononQHA(st_dict_eq, supercell_matrix, properties=prop)
        qha.run(eps_min=0.8, eps_max=1.2, eps_int=0.02,
                mesh=[10,10,10], t_min=100, t_max=1000, t_step=10, disp=0.01)
    elif run_qha and ph.is_imaginary():
        print('Phonon QHA is not performed '
              'because imaginary modes are detected.')
 
    return 0

   
if __name__ == '__main__':

    import argparse
    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar', 
                        type=str, 
                        default=None,
                        help='poscar file')
    parser.add_argument('--pot', 
                        nargs='*',
                        type=str, 
                        default='polymlp.lammps',
                        help='polymlp file')
    args = parser.parse_args()

    prop = Properties(pot=args.pot)
    unitcell = Poscar(args.poscar).get_structure()

    run_single_structure(unitcell, properties=prop, run_qha=True)
