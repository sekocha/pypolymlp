#!/usr/bin/env python 
import numpy as np
import signal

from pypolymlp.api.pypolymlp_fc import PypolymlpFC

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    polymlp_fc = PypolymlpFC('polymlp.lammps')

    #yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
    #polymlp_fc.compute_fcs_phono3py_yaml(yaml)

    unitcell_dict = polymlp_fc.parse_poscar('POSCAR-unitcell')
    supercell_matrix = np.diag([3,3,2])
    polymlp_fc.compute_fcs(unitcell_dict=unitcell_dict,
                           supercell_matrix=supercell_matrix,
                           n_samples=1000,
                           displacements=0.03)


