#!/usr/bin/env python

from pypolymlp.calculator.compute_fcs import (
    compute_fcs_from_structure,
    compute_fcs_phono3py_dataset,
)
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import load_mlp_lammps
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_st_dict,
    st_dict_to_phonopy_cell,
)


class PypolymlpFC:

    def __init__(self, pot):
        """
        Parameters
        ----------
        pot: polymlp.lammps file name
        """

        self._params_dict, mlp_dict = load_mlp_lammps(filename=pot)
        self._params_dict["element_swap"] = False
        self._coeffs = mlp_dict["coeffs"] / mlp_dict["scales"]

        self.unitcell_dict = None
        self.supercell_dict = None

    def parse_poscar(self, poscar):
        st_dict = Poscar(poscar).get_structure()
        return st_dict

    def phonopy_cell_to_pypolymlp_str(self, cell):
        return phonopy_cell_to_st_dict(cell)

    def pypolymlp_str_to_phonopy_cell(self, st_dict):
        return st_dict_to_phonopy_cell(st_dict)

    def compute_fcs(
        self,
        unitcell_dict=None,
        supercell_matrix=None,
        supercell_dict=None,
        n_samples=500,
        displacements=0.03,
        is_plusminus=False,
    ):
        """
        Parameters
        ----------
        unitcell_dict:
        supercell_matrix:
        supercell_dict:
        Set of (unitcell_dict and supercell_matrix)
        or supercell_dict is required.

        n_samples: Number of structures used for force calculations
            using polymlp.
        displacements: Random displacement (in Angstrom)

        Return
        ------
        fc2.hdf5 and fc3.hdf5
        """
        compute_fcs_from_structure(
            params_dict=self._params_dict,
            coeffs=self._coeffs,
            unitcell_dict=unitcell_dict,
            supercell_matrix=supercell_matrix,
            supercell_dict=supercell_dict,
            n_samples=n_samples,
            displacements=displacements,
            is_plusminus=is_plusminus,
        )

    def compute_fcs_phono3py_yaml(
        self,
        phono3py_yaml,
        n_samples=None,
        displacements=0.03,
        is_plusminus=False,
    ):
        """
        Parameters
        ----------
        phono3py_yaml: phono3py yaml.xz file
        n_samples: Number of structures with random displacements.
            Forces acting on atoms in the structures are calculated
            using polymlp. If n_samples = None, all displacements
            included in phono3py yaml file are used for calculating forces
            using polymlp.
        displacements: Random displacement (in Angstrom)

        Return
        ------
        fc2.hdf5 and fc3.hdf5
        """

        compute_fcs_phono3py_dataset(
            params_dict=self._params_dict,
            coeffs=self._coeffs,
            phono3py_yaml=phono3py_yaml,
            use_phonon_dataset=False,
            n_samples=n_samples,
            displacements=displacements,
            is_plusminus=is_plusminus,
        )

    @property
    def parameters(self):
        return self._params_dict
