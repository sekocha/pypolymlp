#!/usr/bin/env python
import numpy as np
import os
import yaml

import phono3py

def parse_dataset():

    yml = yaml.safe_load(open('cells.yaml'))
    supercell = yml['supercell']
    supercell_mat = supercell['supercell_matrix']
    supercell['axis'] = np.array(supercell['axis']).T
    supercell['positions'] = np.array(supercell['positions']).T

    if os.path.exists('data_disps.npy'):
        disps = np.load('data_disps.npy')
        forces = np.load('data_forces.npy')
    else:
        yml = yaml.safe_load(open('samples.yaml'))
        samples = yml['structure_samples']
        disps = np.array([samp['disps_cartesian'] for samp in samples])
        forces = np.array([samp['forces'] for samp in samples])
        np.save('data_disps.npy', disps)
        np.save('data_forces.npy', forces)

    n_data = disps.shape[0]
    disps = disps.reshape((n_data,-1))
    forces = forces.reshape((n_data,-1))
    return disps, forces, supercell

def parse_dataset_phono3py_xz(filename):

    ph = phono3py.load(filename, produce_fc=False, log_level=1)
    supercell_phonopy = ph.supercell

    n_data = ph.dataset["displacements"].shape[0]
    disps = ph.dataset["displacements"].reshape((n_data,-1))
    forces = ph.dataset["forces"].reshape((n_data,-1))
    return disps, forces, supercell_phonopy


