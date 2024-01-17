#!/usr/bin/env python
import numpy as np
import sys

from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_io import Restart, save_cell


if __name__ == '__main__':

    res_yaml = sys.argv[1]
    fc2hdf5 = sys.argv[2]
    pot = sys.argv[3]

    res = Restart(res_yaml, fc2hdf5=fc2hdf5)
    params_dict, coeffs = res.get_mlp_properties(pot=pot)

    ph_real = HarmonicReal(res.supercell,
                           params_dict,
                           coeffs,
                           n_unitcells=res.n_unitcells,
                           fc2=res.force_constants)

    n_samples = 1000
    ph_real.run(t=res.temperature, n_samples=n_samples)

    disps = ph_real.displacements.transpose((0,2,1))
    forces = ph_real.forces.transpose((0,2,1))

    f = open('sscha_cells.yaml', 'a')
    save_cell(res.unitcell, tag='unitcell', fstream=f)
    save_cell(ph_real.supercell, tag='supercell', fstream=f)
    f.close()

    np.save('sscha_disps.npy', disps)
    np.save('sscha_forces.npy', forces)
    print('Tempearture:', res.temperature)
    print('Number of structures:', n_samples)
    print('sscha_disps.npy and sscha_forces.npy are generated.')
    print('- shape:', forces.shape)

