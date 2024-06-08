#!/usr/bin/env python
import signal
import sys

import numpy as np

from pypolymlp.api.pypolymlp import Pypolymlp


def parse_yaml(yaml_file, energy_dat):

    ph3 = Phono3pyYaml(yaml_file)
    disps, forces = ph3.get_phonon_dataset()
    st_dict, _ = ph3.get_structure_dataset()
    energies = np.loadtxt(energy_dat)[1:, 1]
    return disps, forces, energies, st_dict


def combine_array(array1, array2):
    array = []
    for d in array1:
        array.append(d)
    for d in array2:
        array.append(d)
    return np.array(array)


if __name__ == "__main__":
    from pypolymlp.core.interface_phono3py import Phono3pyYaml

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    n_str1 = round(int(sys.argv[1]) * 0.9)
    n_str2 = int(sys.argv[1]) - n_str1

    polymlp = Pypolymlp()
    polymlp.set_params(
        ["Ag", "I"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[8, 8],
        reg_alpha_params=[-2, 2, 15],
        gaussian_params2=[1.0, 7.0, 7],
        # gaussian_params2=[1.0,7.0,12],
        atomic_energy=[-0.23677472, -0.18136690],
    )

    dir_prefix = "/home/seko/collaboration/togo/RS-ZB-WZ-ltc-data-ver4/"
    yaml_file1 = (
        dir_prefix
        + "zincblende/phono3py_params_zincblende_yaml/"
        + "phono3py_params_zincblende_AgI.yaml.xz"
    )
    energy_dat1 = (
        dir_prefix
        + "zincblende/energies_ltc_zincblende_fc3-forces/"
        + "energies_ltc_zincblende_AgI_fc3-forces.dat"
    )
    yaml_file2 = (
        dir_prefix
        + "zincblende-d01/phono3py_params_zincblende_yaml/"
        + "phono3py_params_zincblende_AgI.yaml.xz"
    )
    energy_dat2 = (
        dir_prefix
        + "zincblende-d01/energies_ltc_zincblende_fc3-forces/"
        + "energies_ltc_zincblende_AgI_fc3-forces.dat"
    )

    disps1, forces1, energies1, st_dict = parse_yaml(yaml_file1, energy_dat1)
    disps2, forces2, energies2, _ = parse_yaml(yaml_file2, energy_dat2)

    train_disps = combine_array(disps1[:n_str1], disps2[:n_str2])
    train_forces = combine_array(forces1[:n_str1], forces2[:n_str2])
    train_energies = combine_array(energies1[:n_str1], energies2[:n_str2])

    # test_disps = combine_array(disps1[390:400], disps2[390:400])
    # test_forces = combine_array(forces1[390:400], forces2[390:400])
    # test_energies = combine_array(energies1[390:400], energies2[390:400])
    test_disps = disps1[380:400]
    test_forces = forces1[380:400]
    test_energies = energies1[380:400]

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
