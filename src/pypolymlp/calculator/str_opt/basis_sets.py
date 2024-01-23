#!/usr/bin/env python 
import numpy as np

from typing import Optional
        
import scipy
from scipy.sparse import csr_array

from symfc.spg_reps import SpgRepsO1

from symfc.utils.eig_tools import eigsh_projector

from symfc.utils.matrix_tools_O1 import compressed_projector_sum_rules
from symfc.utils.utils import SymfcAtoms
from symfc.utils.utils_O1 import (
    get_compr_coset_reps_sum,
    get_lat_trans_compr_matrix,
    get_lat_trans_decompr_indices,
)

from symfc.basis_sets.basis_sets_base import FCBasisSetBase

class FCBasisSetO1Base(FCBasisSetBase):
    """Base class of FCBasisSetO1."""

    def __init__(
        self,
        supercell: SymfcAtoms,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        use_mkl : bool
            Use MKL or not. Default is False.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO1(supercell)

    def _get_c_trans(self) -> csr_array:
        trans_perms = self._spg_reps.translation_permutations
        n_lp, N = trans_perms.shape
        decompr_idx = get_lat_trans_decompr_indices(trans_perms)
        c_trans = get_lat_trans_compr_matrix(decompr_idx, N, n_lp)
        return c_trans

class FCBasisSetO1(FCBasisSetO1Base):
    """Dense symmetry adapted basis set for 1st order force constants.

    Attributes
    ----------
    basis_set : ndarray
        Compressed force constants basis set. The first dimension n_x (< n_a) is
        given as a result of compression, which is depends on the system.
        shape=(n_x * N * 9, n_bases), dtype='double'
    full_basis_set : ndarray
        Full (decompressed) force constants basis set. shape=(N * N * 9,
        n_bases), dtype='double'
    translation_permutations : ndarray
        Atom indices after lattice translations. shape=(lattice_translations,
        supercell_atoms), dtype=int.

    """

    def __init__(
        self,
        supercell: SymfcAtoms,
        use_mkl: bool = False,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        supercell : SymfcAtoms
            Supercell.
        log_level : int, optional
            Log level. Default is 0.

        """
        super().__init__(supercell, use_mkl=use_mkl, log_level=log_level)
        self._spg_reps = SpgRepsO1(supercell)
        self._n_a_compression_matrix: Optional[csr_array] = None

    @property
    def basis_set(self) -> Optional[np.ndarray]:
        return self._basis_set

    @property
    def full_basis_set(self) -> Optional[np.ndarray]:
        raise KeyError('Not available.')

    def run(self): # -> FCBasisSetO1:
        raise KeyError('Not available.')


def get_fc_basis_O1(supercell):

    n_atom = len(supercell.numbers)
    basis = FCBasisSetO1(supercell)
    spg_reps = basis._spg_reps
    c_trans = basis._get_c_trans()
    coset_reps_sum = get_compr_coset_reps_sum(spg_reps)
    proj_rt = coset_reps_sum @ c_trans.T

    if len(proj_rt.data) > 0:
        c_rt = eigsh_projector(proj_rt)

        proj = compressed_projector_sum_rules(c_rt, n_atom)
        eigvecs = eigsh_projector(proj)
        basis_set = c_rt @ eigvecs
        return basis_set.toarray()
    return []
    

if __name__ == '__main__':

    import argparse
    from pypolymlp.core.interface_vasp import Poscar
    from pypolymlp.utils.phonopy_utils import st_dict_to_phonopy_cell

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--poscar',
                        type=str,
                        default=None,
                        help='poscar file')
    parser.add_argument('--pot',
                        type=str,
                        default='polymlp.lammps',
                        help='polymlp file')
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
    unitcell_ph = st_dict_to_phonopy_cell(unitcell)

    basis_set = get_fc_basis_O1(unitcell_ph)
    print(basis_set)


