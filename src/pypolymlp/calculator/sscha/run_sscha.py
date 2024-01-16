#!/usr/bin/env python 
import numpy as np
import os

from phonopy import Phonopy
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.solvers.solver_O2 import run_solver_dense_O2

from pypolymlp.core.utils import ev_to_kjmol
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.utils.phonopy_utils import (
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)

from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.io import save_sscha_results

from phono3py.file_IO import write_fc2_to_hdf5, read_fc2_from_hdf5

'''
const_bortzmann = 1.380649e-23 # J K^-1
const_bortzmann_ev = 8.617333262e-5 # eV K^-1
'''

class PolymlpSSCHA:

    def __init__(self, 
                 unitcell_dict, 
                 supercell_matrix, 
                 pot=None, 
                 params_dict=None,
                 coeffs=None):

        if pot is not None:
            self.params_dict, mlp_dict = load_mlp_lammps(filename=pot)
            self.coeffs = mlp_dict['coeffs'] / mlp_dict['scales']
        else:
            self.params_dict = params_dict
            self.coeffs = coeffs

        unitcell = st_dict_to_phonopy_cell(unitcell_dict)
        self.phonopy = Phonopy(unitcell, supercell_matrix)
        self.supercell = self.phonopy.supercell
        self.n_unitcells = int(round(np.linalg.det(supercell_matrix)))

        self.supercell_dict = phonopy_cell_to_st_dict(self.supercell)
        self.supercell_dict['masses'] = self.supercell.masses
        self.supercell_dict['supercell_matrix'] = supercell_matrix
        self.supercell_dict['n_unitcells'] = self.n_unitcells

        self.n_atom = len(self.supercell.masses)
        self.fc2_basis = FCBasisSetO2(self.supercell, use_mkl=False).run()

        ''' for bubble diagram
        fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
        compress_mat_fc3 = fc3_basis.compression_matrix
        compress_eigvecs_fc3 = fc3_basis.basis_set
        '''

        self.ph_real = HarmonicReal(self.supercell_dict, 
                                    self.params_dict,
                                    self.coeffs)
        self.ph_recip = HarmonicReciprocal(self.phonopy, 
                                           self.params_dict, 
                                           self.coeffs)
        self.fc2 = None
        self.__sscha_dict = None
        self.__log_dict = None
       
    def __recover_fc2(self, coefs):
        compress_mat = self.fc2_basis.compression_matrix
        compress_eigvecs = self.fc2_basis.basis_set
        fc2 = compress_eigvecs @ coefs
        fc2 = (compress_mat @ fc2).reshape((self.n_atom, self.n_atom, 3, 3))
        return fc2

    def __run_solver_fc2(self):
        ''' Input parameter shapes are different in FCSolverO2 
            and run_solver_fc2.
            FCSolverO2: (n_samples, n_atom, 3)
            run_solver_dense_O2: (n_samples, n_atom * 3)
        '''

        '''
        disps: (n_samples, n_atom, 3)
        forces: (n_samples, n_atom, 3)
        '''
        disps = self.ph_real.displacements.transpose((0,2,1))
        forces = self.ph_real.forces.transpose((0,2,1))
        n_samples = disps.shape[0]

        fc2_coeffs = run_solver_dense_O2(disps.reshape((n_samples, -1)), 
                                         forces.reshape((n_samples, -1)), 
                                         self.fc2_basis.compression_matrix, 
                                         self.fc2_basis.basis_set)
        fc2 = self.__recover_fc2(fc2_coeffs)
        return fc2

    def __unit_kjmol(self, e):
        return ev_to_kjmol(e) / self.n_unitcells

    def __compute_sscha_properties(self, 
                                   t=1000, 
                                   qmesh=[10,10,10], 
                                   first_order=True):

        self.ph_recip.force_constants = self.fc2
        self.ph_recip.compute_thermal_properties(t=t, qmesh=qmesh)

        res_dict = {
            'temperature': t,
            'harmonic_free_energy': self.ph_recip.free_energy, # kJ/mol
            'static_potential': 
                self.__unit_kjmol(self.ph_real.static_potential),
            'harmonic_potential': 
                self.__unit_kjmol(self.ph_real.average_harmonic_potential),
            'average_potential': 
                self.__unit_kjmol(self.ph_real.average_full_potential),
            'anharmonic_free_energy': 
                self.__unit_kjmol(self.ph_real.average_anharmonic_potential),
        }

        if first_order:
            res_dict['anharmonic_free_energy_exact'] = 0.0
            res_dict['free_energy'] = res_dict['harmonic_free_energy'] \
                                    + res_dict['anharmonic_free_energy']
        else:
            pass

        return res_dict

    def __single_iter(self, t=1000, n_samples=100, qmesh=[10,10,10]):

        self.ph_real.force_constants = self.fc2
        self.ph_real.run(t=t, n_samples=n_samples, eliminate_outliers=True)

        self.__sscha_dict = self.__compute_sscha_properties(t=t, qmesh=qmesh)
        fc2 = self.__run_solver_fc2()
        return fc2

    def __convergence_score(self, fc2_init, fc2_update):
        norm1 = np.linalg.norm(fc2_update - fc2_init)
        norm2 = np.linalg.norm(fc2_init)
        return norm1 / norm2

    def set_initial_force_constants(self, algorithm='harmonic', filename=None):
        if algorithm == 'harmonic':
            print('Initial FCs: Harmonic')
            self.fc2 = self.ph_recip.produce_harmonic_force_constants()
        elif algorithm == 'const':
            print('Initial FCs: Constants')
            n_coeffs = self.fc2_basis.basis_set.shape[1]
            coeffs_fc2 = np.ones(n_coeffs) * 10
            coeffs_fc2[1::2] *= -1
            self.fc2 = self.__recover_fc2(coeffs_fc2)
        elif algorithm == 'random':
            print('Initial FCs: Random')
            n_coeffs = self.fc2_basis.basis_set.shape[1]
            coeffs_fc2 = (np.random.rand(n_coeffs) - 0.5) * 20
            self.fc2 = self.__recover_fc2(coeffs_fc2)
        elif algorithm == 'file':
            print('Initial FCs: File', filename)
            self.fc2 = read_fc2_from_hdf5(filename)

    def run(self, 
            t=1000, n_samples=100, qmesh=[10,10,10],
            n_loop=100, tol=1e-2, mixing=0.5, log=True):

        if self.fc2 is None:
            self.fc2 = self.ph_recip.produce_harmonic_force_constants()

        n_iter, delta = 1, 100
        while n_iter <= n_loop and delta > tol:
            if log:
                print('------------- Iteration :', n_iter, '-------------')

            fc2_update = self.__single_iter(t=t, 
                                            n_samples=n_samples, 
                                            qmesh=qmesh)
            delta = self.__convergence_score(self.fc2, fc2_update)
            self.fc2 = fc2_update * mixing + self.fc2 * (1 - mixing)
            n_iter += 1

            if log:
                self.__print_progress(delta)

        converge = True if delta < tol else False
        self.__log_dict = {
            'converge': converge,
            'delta': delta
        }

    def __print_progress(self, delta):

        print('convergence score:         ', "{:.6f}".format(delta))
        print('thermodynamic_properties:')
        print('- free energy (harmonic)  :',
            "{:.6f}".format(self.__sscha_dict['harmonic_free_energy']), 
            '(kJ/mol)')
        print('  free energy (anharmonic):',
            "{:.6f}".format(self.__sscha_dict['anharmonic_free_energy']), 
            '(kJ/mol)')
        print('  free energy (sscha)     :',
            "{:.6f}".format(self.__sscha_dict['free_energy']), 
            '(kJ/mol)')

    @property
    def properties(self):
        return self.__sscha_dict

    @property
    def logs(self):
        return self.__log_dict

    @property
    def force_constants(self):
        return self.fc2

    @force_constants.setter
    def force_constants(self, fc2):
        ''' (n_atom, n_atom, 3, 3)'''
        self.fc2 = fc2


def run_sscha(unitcell_dict, 
              supercell_matrix,
              args,
              pot=None,
              params_dict=None,
              coeffs=None,
              log=True):

    sscha = PolymlpSSCHA(unitcell_dict, 
                         supercell_matrix, 
                         pot=pot, 
                         params_dict=params_dict, 
                         coeffs=coeffs)

    sscha.set_initial_force_constants(algorithm=args.init, 
                                      filename=args.init_file)

    for temp in args.temperatures:
        print('************** Temperature:', temp, '**************')
        sscha.run(t=temp, 
                  n_samples=args.n_steps,
                  qmesh=args.mesh,
                  n_loop=args.max_iter,
                  tol=args.tol,
                  mixing=args.mixing)
        print('Increasing number of samples.')
        sscha.run(t=temp, 
                  n_samples=args.n_steps_final,
                  qmesh=args.mesh,
                  n_loop=args.max_iter,
                  tol=args.tol,
                  mixing=args.mixing)

        ''' file output'''
        log_dir = './sscha/' + str(temp) + '/'
        os.makedirs(log_dir, exist_ok=True)
        save_sscha_results(sscha.properties, 
                           sscha.logs, 
                           unitcell_dict, 
                           supercell_matrix,
                           args,
                           filename=log_dir + 'sscha_results.yaml')
        write_fc2_to_hdf5(sscha.force_constants, filename=log_dir + 'fc2.hdf5')


if __name__ == '__main__':

    import argparse
    import signal
    from pypolymlp.core.interface_vasp import Poscar
    from pypolymlp.calculator.sscha.io import (
            temperature_setting,
            n_steps_setting,
            print_parameters,
            print_structure
    )

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar', 
                        type=str, 
                        default='POSCAR',
                        help='poscar file (unit cell)')
    parser.add_argument('--pot', 
                        type=str, 
                        default='polymlp.lammps',
                        help='polymlp.lammps file')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')
    parser.add_argument('--mesh',
                        type=int,
                        nargs=3,
                        default=[10,10,10],
                        help='k-mesh used for phonon calculation')
    parser.add_argument('-t', '--temp',
                        type=float,
                        default=None,
                        help='Temperature (K)')
    parser.add_argument('-t_min', '--temp_min',
                        type=float,
                        default=100,
                        help='Temperature to begin (K)')
    parser.add_argument('-t_max', '--temp_max',
                        type=float,
                        default=2000,
                        help='Temperature to end (K)')
    parser.add_argument('-t_step', '--temp_step',
                        type=float,
                        default=100,
                        help='Temperature interval (K)')
    parser.add_argument('--tol',
                        type=float,
                        default=0.005,
                        help='Tolerance parameter for FC convergence')
    parser.add_argument('--n_samples',
                        type=int,
                        nargs=2,
                        default=None,
                        help='Number of steps used in ' +
                             'iterations and the last iteration')
    parser.add_argument('--max_iter',
                        type=int,
                        default=50,
                        help='Maximum number of iterations')
    parser.add_argument('--ascending_temp',
                        action='store_true',
                        help='use ascending order of temperatures')
    parser.add_argument('--init',
                        choices=['harmonic','const','random','file'],
                        default='harmonic',
                        help='initial FCs')
    parser.add_argument('--init_file',
                        default=None,
                        help='file name for initial FCs')
    parser.add_argument('--mixing',
                        type=float,
                        default=0.5,
                        help='mixing')
    args = parser.parse_args()


    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)

    n_atom_supercell = len(unitcell_dict['elements']) \
                     * np.linalg.det(supercell_matrix)

    args = temperature_setting(args)
    args = n_steps_setting(args, n_atom_supercell)

    print_parameters(supercell_matrix, args)
    print_structure(unitcell_dict)

    run_sscha(unitcell_dict, supercell_matrix, args, pot=args.pot)


