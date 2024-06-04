#!/usr/bin/env python 
import numpy as np
import signal
import glob
import sys

from pypolymlp.api.pypolymlp import Pypolymlp
from pypolymlp.core.io_polymlp import save_mlp_lammps

def parse_yaml(yaml_file, energy_dat):
    
    ph3 = Phono3pyYaml(yaml_file)
    disps, forces = ph3.get_phonon_dataset()
    st_dict, _ = ph3.get_structure_dataset()
    energies = np.loadtxt(energy_dat)[1:,1]
    return disps, forces, energies, st_dict

if __name__ == '__main__':
    from pypolymlp.core.interface_phono3py import Phono3pyYaml

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    n_str = int(sys.argv[1])

    polymlp = Pypolymlp()
    polymlp.set_params(
            ['Ag','I'],
            cutoff=8.0,
            model_type=3,
            max_p=2,
            gtinv_order=3,
            gtinv_maxl=[8,8],
            reg_alpha_params=[-2,2,15],
            gaussian_params2=[1.0,7.0,7],
            atomic_energy=[-0.23677472,-0.18136690],
    )


    dir_prefix = '/home/seko/collaboration/togo/RS-ZB-WZ-ltc-data-ver5/'
    yaml_file = dir_prefix + 'zincblende-2000/phono3py_params_zincblende_yaml/phono3py_params_zincblende_AgI.yaml.xz'
    energy_dat = dir_prefix + 'zincblende-2000/energies_ltc_zincblende_fc3-forces/energies_ltc_zincblende_AgI_fc3-forces.dat'

    disps, forces, energies, st_dict = parse_yaml(yaml_file, energy_dat)

    coeffs_all = []
    n_loops = 20
    for i in range(n_loop):
        #n_bagging = int(n_str * 0.8)
        #str_ids = np.random.choice(n_str, n_bagging)
        str_ids = np.random.choice(n_str, n_str)
        print(str_ids)
        
        train_disps = disps[str_ids]
        train_forces = forces[str_ids]
        train_energies = energies[str_ids]

        test_disps = disps[1980:2000]
        test_forces = forces[1980:2000]
        test_energies = energies[1980:2000]

        polymlp.set_datasets_displacements(
            train_disps,
            train_forces,
            train_energies,
            test_disps,
            test_forces,
            test_energies,
            st_dict,
        )
        polymlp.run(log=True)

        params_dict = polymlp.parameters
        mlp_dict = polymlp.summary
        coeffs = mlp_dict['coeffs'] / mlp_dict['scales']
        coeffs_all.append(coeffs)

    coeffs_ave = np.average(np.array(coeffs_all), axis=0)
    scales_ave = np.ones(coeffs_ave.shape[0])

    save_mlp_lammps(params_dict, coeffs_ave, scales_ave)



