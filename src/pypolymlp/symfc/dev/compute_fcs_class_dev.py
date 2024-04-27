#!/usr/bin/env python 
import numpy as np
import time
import gc

from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.utils.phonopy_utils import (
        phonopy_supercell,
        phonopy_cell_to_st_dict,
        st_dict_to_phonopy_cell,
)
from pypolymlp.core.displacements import (
        generate_random_const_displacements,
        get_structures_from_displacements,
)

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym


import phonopy
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5


from symfc.spg_reps import SpgRepsO1
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.solvers.solver_O2O3 import (
    run_solver_O2O3,
    run_solver_O2O3_no_sum_rule_basis,
)
from symfc.utils.utils_O3 import get_lat_trans_compr_matrix_O3
from symfc.utils.matrix_tools_O3 import set_complement_sum_rules


'''symfc_basis_dev: must be included to FCBasisSetO3 in symfc'''
from pypolymlp.symfc.dev.symfc_basis_dev import run_basis


def recover_fc2(coefs, compress_mat, compress_eigvecs, N):
    n_a = compress_mat.shape[0] // (9*N)
    n_lp = N // n_a
    fc2 = compress_eigvecs @ coefs
    fc2 = (compress_mat @ fc2).reshape((N,N,3,3))
    fc2 /= np.sqrt(n_lp)
    return fc2


def recover_fc3(coefs, compress_mat, compress_eigvecs, N):
    n_a = compress_mat.shape[0] // (27*(N**2))
    n_lp = N // n_a
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((n_a,N,N,3,3,3))
    fc3 /= np.sqrt(n_lp)
    return fc3


def recover_fc3_variant(
    coefs, compress_mat, proj_pt, trans_perms, n_iter=10,
):
    ''' if using full compression_matrix
    fc3 = compress_eigvecs @ coefs
    fc3 = (compress_mat @ fc3).reshape((N,N,N,3,3,3))
    '''
    n_lp, N = trans_perms.shape
    n_a = compress_mat.shape[0] // (27*(N**2))

    fc3 = compress_mat @ coefs
    c_sum_cplmt = set_complement_sum_rules(trans_perms)

    for i in range(n_iter):
        fc3 -= c_sum_cplmt.T @ (c_sum_cplmt @ fc3)
        fc3 = proj_pt @ fc3

    fc3 = fc3.reshape((n_a,N,N,3,3,3))
    fc3 /= np.sqrt(n_lp)
    return fc3




class PolymlpFC:

    def __init__(self, 
                 supercell=None, 
                 phono3py_yaml=None,
                 use_phonon_dataset=False,
                 pot=None, 
                 params_dict=None, 
                 coeffs=None,
                 properties=None):

        '''
        Parameters
        ----------
        supercell: Supercell in phonopy format or structure dict
        pot, (params_dict and coeffs), or Properties object: polynomal MLP
        '''

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, 
                                   params_dict=params_dict, 
                                   coeffs=coeffs)

        self.__initialize_supercell(supercell=supercell,
                                    phono3py_yaml=phono3py_yaml,
                                    use_phonon_dataset=use_phonon_dataset)


    def __initialize_supercell(self,
                               supercell=None,
                               phono3py_yaml=None,
                               use_phonon_dataset=False):

        if supercell is not None:
            if isinstance(supercell, dict):
                self.__supercell_dict = supercell
                self.__supercell_ph = st_dict_to_phonopy_cell(supercell)
            elif isinstance(supercell, phonopy.structure.cells.Supercell):
                self.__supercell_dict = phonopy_cell_to_st_dict(supercell)
                self.__supercell_ph = supercell
            else:
                raise ValueError('PolymlpFC: type(supercell) must be'
                                 ' dict or phonopy supercell')

        elif phono3py_yaml is not None:
            (self.__supercell_ph, self.__disps, self.__st_dicts) \
                = parse_phono3py_yaml_fcs(
                    phono3py_yaml, use_phonon_dataset=use_phonon_dataset
                )
            self.__supercell_dict = phonopy_cell_to_st_dict(self.__supercell_ph)

        else:
            raise ValueError('PolymlpFC: supercell or phonon3py_yaml'
                             ' is required for initialization')

        self.__N = len(self.__supercell_ph.symbols)
        return self


    def sample(self, n_samples=100, displacements=0.001, is_plusminus=False):

        if n_samples is not None:
            self.__disps, self.__st_dicts \
                = generate_random_const_displacements(
                    self.__supercell_dict,
                    n_samples=n_samples,
                    displacements=displacements,
                    is_plusminus=is_plusminus,
                  )
        return self

    def run_geometry_optimization(self, gtol=1e-6):

        print('Running geometry optimization')
        try:
            minobj = MinimizeSym(self.__supercell_dict, properties=self.prop)
        except:
            print('No geomerty optimization is performed.')
            return self

        minobj.run(gtol=gtol)
        print('Residual forces:')
        print(minobj.residual_forces.T)
        print('E0:', minobj.energy)
        print('n_iter:', minobj.n_iter)
        print('Fractional coordinate changes:')
        diff_positions = self.__supercell_dict['positions'] \
                            - minobj.structure['positions']
        print(diff_positions.T)
        print('Success:', minobj.success)

        if minobj.success:
            self.__supercell_dict = minobj.structure
            self.__supercell_ph = st_dict_to_phonopy_cell(self.__supercell_dict)
            self.__st_dicts = get_structures_from_displacements(
                self.__disps, self.__supercell_dict
            )

        return self

    def __compute_forces(self):

        _, forces, _ = self.prop.eval_multiple(self.__st_dicts)
        _, residual_forces, _ = self.prop.eval(self.__supercell_dict)
        for f in forces:
            f -= residual_forces
        return forces

    def run(self, forces=None, batch_size=100, sum_rule_basis=True):

        if forces is None:
            print('Computing forces using polymlp')
            t1 = time.time()
            forces = self.__compute_forces()
            t2 = time.time()
            print(' elapsed time (computing forces) =', t2-t1)

        ''' 
        disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        '''
        disps = self.__disps.transpose((0,2,1)) 
        forces = np.array(forces).transpose((0,2,1)) 

        n_data, N, _ = forces.shape
        disps = disps.reshape((n_data, -1))
        forces = forces.reshape((n_data, -1))

        ''' Constructing fc2 basis and fc3 basis '''
        t1 = time.time()
        fc2_basis = FCBasisSetO2(self.__supercell_ph, use_mkl=False).run()
        compress_mat_fc2_full = fc2_basis.compression_matrix
        compress_eigvecs_fc2 = fc2_basis.basis_set

        if sum_rule_basis:
            compress_mat_fc3, compress_eigvecs_fc3 = run_basis(
                self.__supercell_ph, apply_sum_rule=True,
            )
        else:
            compress_mat_fc3, proj_pt = run_basis(self.__supercell_ph, 
                                                  apply_sum_rule=False)

        trans_perms = SpgRepsO1(self.__supercell_ph).translation_permutations
        c_trans = get_lat_trans_compr_matrix_O3(trans_perms)
        compress_mat_fc3_full = c_trans @ compress_mat_fc3
        del c_trans
        gc.collect()

        t2 = time.time()
        print(' elapsed time (basis sets for fc2 and fc3) =', t2-t1)


        print('----- Solving fc2 and fc3 using run_solver -----')
        t1 = time.time()
        use_mkl = False if N > 400 else True
        if sum_rule_basis:
            coefs_fc2, coefs_fc3 = run_solver_O2O3(
                disps, forces, 
                compress_mat_fc2_full, compress_mat_fc3_full, 
                compress_eigvecs_fc2, compress_eigvecs_fc3,
                use_mkl=use_mkl,
                batch_size=batch_size,
            )
        else:
            coefs_fc2, coefs_fc3 = run_solver_O2O3_no_sum_rule_basis(
                disps, forces, 
                compress_mat_fc2_full, compress_mat_fc3_full, 
                compress_eigvecs_fc2,
                use_mkl=use_mkl,
                batch_size=batch_size
            )
        t2 = time.time()
        print(' elapsed time (solve fc2 + fc3) =', t2-t1)

        t1 = time.time()
        fc2 = recover_fc2(
            coefs_fc2, compress_mat_fc2_full, compress_eigvecs_fc2, self.__N
        )

        if sum_rule_basis:
            fc3 = recover_fc3(coefs_fc3, compress_mat_fc3, 
                              compress_eigvecs_fc3, self.__N)
        else:
            print('Applying sum rules to fc3')
            fc3 = recover_fc3_variant(
                coefs_fc3, compress_mat_fc3, proj_pt, trans_perms
            )

        t2 = time.time()
        print(' elapsed time (recover fc2 and fc3) =', t2-t1)

        print('writing fc2.hdf5') 
        write_fc2_to_hdf5(fc2)
        print('writing fc3.hdf5') 
        write_fc3_to_hdf5(fc3)

        return self

    @property
    def displacements(self):
        return self.__disps

    @property
    def structures(self):
        return self.__st_dicts

    @displacements.setter
    def displacements(self, disps):
        '''disps: Displacements (n_str, 3, n_atom)'''
        if not disps.shape[1] == 3 or not disps.shape[2] == self.__N:
            raise ValueError('displacements must have a shape of '
                             '(n_str, 3, n_atom)')
        self.__disps = disps

    @structures.setter
    def structures(self, st_dicts):
        self.__st_dicts = st_dicts



if __name__ == '__main__':

    import argparse
    import signal
    from pypolymlp.core.interface_vasp import Poscar
    #from pypolymlp.utils.yaml_utils import load_cells

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--poscar',
                        type=str,
                        default=None,
                        help='poscar')
    parser.add_argument('--supercell',
                        nargs=3,
                        type=int,
                        default=None,
                        help='Supercell size (diagonal components)')

    parser.add_argument('--pot',
                        type=str,
                        default=None,
                        help='polymlp file')
    parser.add_argument('--fc_n_samples',
                        type=int,
                        default=None,
                        help='Number of random displacement samples')
    parser.add_argument('--disp',
                        type=float,
                        default=0.03,
                        help='Displacement (in Angstrom)')
    parser.add_argument('--is_plusminus',
                        action='store_true',
                        help='Plus-minus displacements will be generated.')
    parser.add_argument('--geometry_optimization',
                        action='store_true',
                        help='Geometry optimization is performed '
                             'for initial structure.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Batch size for FC solver.')

    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).get_structure()
    supercell_matrix = np.diag(args.supercell)
    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    polyfc = PolymlpFC(supercell=supercell, pot=args.pot)

    if args.fc_n_samples is not None:
        polyfc.sample(n_samples=args.fc_n_samples, 
                      displacements=args.disp, 
                      is_plusminus=args.is_plusminus)
    if args.geometry_optimization:
        polyfc.run_geometry_optimization()

    polyfc.run()