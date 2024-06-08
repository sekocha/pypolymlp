#!/usr/bin/env python
import sys

import phono3py


class Phono3pyYamlTest:

    def __init__(self, yaml_filename):
        """displacements: (n_samples, 3, n_atom)
        forces: (n_samples, 3, n_atom)
        positions_all: (n_samples, 3, n_atom)
        """
        ph3 = phono3py.load(yaml_filename, produce_fc=False, log_level=1)
        self.supercell = ph3.supercell
        self.displacements = ph3.displacements.transpose((0, 2, 1))  # Angstrom
        self.forces = ph3.forces.transpose((0, 2, 1))  # eV/Angstrom
        print(self.forces)
        # print(ph3.phonon_supercell_matrix)
        # print(ph3.phonon_supercell)
        # print(ph3.phonon_dataset['forces'].shape)
        print(ph3.unitcell)

    def get_phonon_dataset(self):
        """displacements: (n_samples, 3, n_atom)
        forces: (n_samples, 3, n_atom)
        """
        return (self.displacements, self.forces)

    def get_structure_dataset(self):
        """positions_all: (n_samples, 3, n_atom)"""
        return (self.st_dict, self.positions_all)


ph3 = Phono3pyYamlTest(sys.argv[1])
