#!/usr/bin/env python
import numpy as np
import sys

from pypolymlp.utils.phonopy_utils import phonopy_supercell
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_io import save_cell, load_restart


if __name__ == '__main__':

    res_yaml = sys.argv[1]
    fc2hdf5 = sys.argv[2]
    pot = sys.argv[3]

    res_sscha, struct, params_dict, coeffs, fc2 = load_restart(res_yaml, 
                                                               fc2hdf5, 
                                                               pot=pot)
    unitcell_dict, supercell_matrix = struct
    supercell_dict = phonopy_supercell(unitcell_dict, 
                                       supercell_matrix=supercell_matrix, 
                                       return_phonopy=False)
    n_unitcells = int(round(np.linalg.det(supercell_matrix)))
    ph_real = HarmonicReal(supercell_dict,
                           params_dict,
                           coeffs,
                           n_unitcells=n_unitcells,
                           fc2=fc2)

    temp = float(res_sscha[0]['temperature'])
    ph_real.run(t=temp, n_samples=1000)

    disps = ph_real.displacements.transpose((0,2,1))
    forces = ph_real.forces.transpose((0,2,1))

    f = open('data_cells.yaml', 'a')
    save_cell(unitcell_dict, tag='unitcell', fstream=f)
    save_cell(supercell_dict, tag='supercell', fstream=f)
    f.close()

    np.save('data_disps.npy', disps)
    np.save('data_forces.npy', forces)
    print('data_disps.npy and data_forces.npy are generated.')
    print('- shape:', forces.shape)

