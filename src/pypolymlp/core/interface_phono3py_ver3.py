"""Interface for phono3py."""

import numpy as np
import phono3py

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpStructure
from pypolymlp.core.displacements import (
    convert_disps_to_positions,
    get_structures_from_multiple_positions,
    set_dft_data,
)
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure


def parse_phono3py_yaml(
    yamlfile,
    energies_filename=None,
    element_order=None,
    select_ids=None,
    return_displacements=False,
    use_phonon_dataset=False,
) -> PolymlpDataDFT:
    """Read phono3py.yaml and return DFT dataclass.

    Parameters
    ----------
    yamlfile: phono3py yaml file
    energies_filename: first line gives the energy of perfect supercell

    phono3py.yaml must be composed of datasets for a fixed composition.
    """

    ph3 = Phono3pyYaml(yamlfile, use_phonon_dataset=use_phonon_dataset)
    disps, forces = ph3.phonon_dataset
    supercell, positions_all = ph3.structure_dataset

    if energies_filename is not None:
        energies = np.loadtxt(energies_filename)[1:, 1]
    else:
        energies = ph3.energies
        if energies is None:
            raise ValueError("No energy entry in phono3py.yaml or no energy file.")

    if select_ids is not None:
        select_ids = np.array(select_ids)
        forces = forces[select_ids]
        positions_all = positions_all[select_ids]
        energies = energies[select_ids]
        disps = disps[select_ids]

    dft = set_dft_data(
        forces,
        energies,
        positions_all,
        supercell,
        element_order=element_order,
    )
    if return_displacements:
        return dft, disps
    return dft


def parse_structures_from_phono3py_yaml(phono3py_yaml, select_ids=None):
    ph3 = Phono3pyYaml(phono3py_yaml)
    if select_ids is not None:
        return [ph3.supercells[i] for i in select_ids]
    return ph3.supercells


def parse_phono3py_yaml_fcs(phono3py_yaml, use_phonon_dataset=False):
    """displacements: shape=(n_str, 3, n_atoms)"""
    ph3 = Phono3pyYaml(phono3py_yaml, use_phonon_dataset=use_phonon_dataset)
    return (ph3.supercell_phono3py, ph3.displacements, ph3.supercells)


class Phono3pyYaml:

    def __init__(self, yamlfile, use_phonon_dataset=False):
        self.__ph3 = phono3py.load(yamlfile, produce_fc=False, log_level=1)

        if use_phonon_dataset is False:
            self.__supercell = self.__ph3.supercell
            self.__displacements = self.__ph3.displacements.transpose((0, 2, 1))
            self.__forces = self.__ph3.forces.transpose((0, 2, 1))
            try:
                self.__energies = self.__ph3.supercell_energies
            except:
                self.__energies = None
        else:
            print("Using phono3py.phonon_*** dataset")
            self.__supercell = self.__ph3.phonon_supercell
            self.__displacements = self.__ph3.phonon_dataset["displacements"]
            self.__displacements = self.__displacements.transpose((0, 2, 1))
            self.__forces = self.__ph3.phonon_dataset["forces"]
            self.__forces = self.__forces.transpose((0, 2, 1))
            """TODO: Must be revised"""
            self.__energies = None

        self.__supercell_polymlp = phonopy_cell_to_structure(self.__supercell)
        self.__positions_all = convert_disps_to_positions(
            self.__displacements,
            self.__supercell_polymlp.axis,
            self.__supercell_polymlp.positions,
        )

    @property
    def phono3py(self):
        return self.__ph3

    @property
    def supercell_phono3py(self):
        return self.__supercell

    @property
    def supercell_polymlp(self):
        return self.__supercell_polymlp

    @property
    def displacements(self):
        """displacements: shape=(n_samples, 3, n_atom)"""
        return self.__displacements

    @property
    def forces(self):
        """forces: shape=(n_samples, 3, n_atom)"""
        return self.__forces

    @property
    def energies(self):
        """forces: shape=(n_samples)"""
        return self.__energies

    @property
    def phonon_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.displacements, self.forces)

    @property
    def structure_dataset(self) -> tuple[PolymlpStructure, np.ndarray]:
        """positions_all: shape=(n_samples, 3, n_atom)"""
        return (self.supercell_polymlp, self.__positions_all)

    def supercells(self) -> list[PolymlpStructure]:
        """Return supercells in PolymlpStructure format."""
        return get_structures_from_multiple_positions(
            self.__positions_all,
            self.supercell_polymlp,
        )
