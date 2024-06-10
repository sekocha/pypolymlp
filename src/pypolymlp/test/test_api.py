#!/usr/bin/env python
import glob
import signal

import numpy as np

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    polymlp = Pypolymlp()

    dataset_type = "vasp"
    #    dataset_type = 'phono3py'
    #    dataset_type = 'displacements'
    if dataset_type == "vasp":
        """from parameters and vasprun.xml files"""
        polymlp.set_params(
            ["Mg", "O"],
            cutoff=8.0,
            model_type=3,
            max_p=2,
            gtinv_order=3,
            gtinv_maxl=[4, 4],
            gaussian_params2=[0.0, 7.0, 8],
            atomic_energy=[-0.00040000, -1.85321219],
        )
        train_vaspruns = glob.glob("vaspruns/train/vasprun-*.xml.polymlp")
        test_vaspruns = glob.glob("vaspruns/test/vasprun-*.xml.polymlp")
        polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
        polymlp.run(verbose=True)

    elif dataset_type == "phono3py":
        """from parameters and phono3py.yaml.xz"""
        polymlp.set_params(
            ["Ag", "I"],
            cutoff=8.0,
            model_type=3,
            max_p=2,
            gtinv_order=3,
            gtinv_maxl=[4, 4],
            gaussian_params2=[0.0, 7.0, 10],
            atomic_energy=[-0.19820116, -0.21203241],
        )
        train_yaml = "phono3py_params_wurtzite_AgI.yaml.xz"
        test_yaml = "phono3py_params_wurtzite_AgI.yaml.xz"
        train_ids = np.arange(20)
        test_ids = np.arange(380, 400)

        polymlp.set_datasets_phono3py(
            train_yaml,
            test_yaml,
            train_ids=train_ids,
            test_ids=test_ids,
        )
        polymlp.run(verbose=True)

    elif dataset_type == "displacements":

        polymlp.set_params(
            ["Ag", "I"],
            cutoff=8.0,
            model_type=3,
            max_p=2,
            gtinv_order=3,
            gtinv_maxl=[4, 4],
            gaussian_params2=[0.0, 7.0, 10],
            atomic_energy=[-0.19820116, -0.21203241],
        )

        """ Parameters (disps, forces, energies, and st_dict) are
            Temporarily obtained from phono3py.yaml.xz
        """
        from pypolymlp.core.interface_phono3py_ver3 import Phono3pyYaml

        yaml_file = "phono3py_params_wurtzite_AgI.yaml.xz"
        energy_dat = "energies_ltc_wurtzite_AgI_fc3-forces.dat"

        ph3 = Phono3pyYaml(yaml_file)
        disps, forces = ph3.phonon_dataset
        st_dict, _ = ph3.structure_dataset
        energies = np.loadtxt(energy_dat)[1:, 1]

        train_disps = disps[0:20]
        train_forces = forces[0:20]
        train_energies = energies[0:20]
        test_disps = disps[380:400]
        test_forces = forces[380:400]
        test_energies = energies[380:400]

        """
        Parameters in polymlp.set_datasets_displacements
        -------------------------------------------------
        train_disps: (n_train, 3, n_atoms)
        train_forces: (n_train, 3, n_atoms)
        train_energies: (n_train)
        test_disps: (n_test, 3, n_atom)
        test_forces: (n_test, 3, n_atom)
        test_energies: (n_test)
        """
        polymlp.set_datasets_displacements(
            train_disps,
            train_forces,
            train_energies,
            test_disps,
            test_forces,
            test_energies,
            st_dict,
        )
        polymlp.run(verbose=True)

    """ from polymlp.in"""
    # polymlp.run(file_params='polymlp.in', log=True)

    """ (params_dict and mlp_dict) or polymlp.lammps file
        can be used for computing properties."""
    params_dict = polymlp.parameters
    mlp_dict = polymlp.summary
