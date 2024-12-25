"""Interface for phono3py."""

from typing import Optional

import numpy as np
import phono3py

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpStructure
from pypolymlp.core.displacements import (
    convert_disps_to_positions,
    get_structures_from_multiple_positions,
)
from pypolymlp.core.interface_datasets import set_dataset_from_structures
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure


def parse_phono3py_yaml(
    yamlfile: str,
    energies_filename: Optional[str] = None,
    element_order: Optional[list[str]] = None,
    select_ids: Optional[list[int]] = None,
    return_displacements: bool = False,
    use_phonon_dataset: bool = False,
) -> PolymlpDataDFT:
    """Read phono3py.yaml and return DFT dataclass.

    Parameters
    ----------
    yamlfile: phono3py yaml file
    energies_filename: Energies.
                       The first line must give the energy of perfect supercell

    phono3py.yaml must be composed of datasets for a fixed composition.
    """

    ph3 = Phono3pyYaml(yamlfile, use_phonon_dataset=use_phonon_dataset)
    disps, forces = ph3.phonon_dataset
    supercell, positions_all = ph3.structure_dataset

    if not np.all(np.isin(element_order, ph3.supercell_phono3py.symbols)):
        raise ValueError("Elements in the input file are not found in phono3py.yaml.")

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

    supercells = get_structures_from_multiple_positions(positions_all, supercell)
    dft = set_dataset_from_structures(
        supercells,
        energies,
        forces=forces,
        stresses=None,
        element_order=element_order,
    )
    if return_displacements:
        return dft, disps
    return dft


def parse_structures_from_phono3py_yaml(
    phono3py_yaml: str, select_ids: Optional[list[int]] = None
) -> list[PolymlpStructure]:
    """Parse phono3py.yaml and return structures."""
    ph3 = Phono3pyYaml(phono3py_yaml)
    if select_ids is not None:
        return [ph3.supercells[i] for i in select_ids]
    return ph3.supercells


def parse_phono3py_yaml_fcs(phono3py_yaml: str, use_phonon_dataset: bool = False):
    """Parse phono3py.yaml and return displacements and structures.

    Return
    ------
    supercell_phono3py: Equilibrium structure.
    displacements: Displacements, shape=(n_str, 3, n_atoms)
    supercells: Supercells with displacements.
    """
    ph3 = Phono3pyYaml(phono3py_yaml, use_phonon_dataset=use_phonon_dataset)
    return (ph3.supercell_phono3py, ph3.displacements, ph3.supercells)


class Phono3pyYaml:
    """Class for phono3py.yaml"""

    def __init__(self, yamlfile: str, use_phonon_dataset: bool = False):
        self._ph3 = phono3py.load(yamlfile, produce_fc=False, log_level=1)
        if use_phonon_dataset == False:
            self._supercell = self._ph3.supercell
            self._displacements = self._ph3.displacements.transpose((0, 2, 1))
            self._forces = self._ph3.forces.transpose((0, 2, 1))
            try:
                self._energies = self._ph3.supercell_energies
            except:
                self._energies = None
        else:
            print("Using phono3py.phonon_*** dataset")
            self._supercell = self._ph3.phonon_supercell
            self._displacements = self._ph3.phonon_dataset["displacements"]
            self._displacements = self._displacements.transpose((0, 2, 1))
            self._forces = self._ph3.phonon_dataset["forces"]
            self._forces = self._forces.transpose((0, 2, 1))
            # TODO: Must be revised
            self._energies = None

        self._supercell_polymlp = phonopy_cell_to_structure(self._supercell)
        self._positions_all = convert_disps_to_positions(
            self._displacements,
            self._supercell_polymlp.axis,
            self._supercell_polymlp.positions,
        )

    @property
    def phono3py(self):
        """Return phono3py object."""
        return self._ph3

    @property
    def supercell_phono3py(self):
        """Return equilibrium supercell in Phonopy format."""
        return self._supercell

    @property
    def supercell_polymlp(self) -> PolymlpStructure:
        """Return equilibrium supercell in PolymlpStructure format."""
        return self._supercell_polymlp

    @property
    def displacements(self) -> np.ndarray:
        """Return displacements, shape=(n_samples, 3, n_atom)."""
        return self._displacements

    @property
    def forces(self) -> np.ndarray:
        """Return forces, shape=(n_samples, 3, n_atom)."""
        return self._forces

    @property
    def energies(self):
        """Retrun energies, shape=(n_samples)."""
        return self._energies

    @property
    def phonon_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Return displacements and forces"""
        return (self.displacements, self.forces)

    @property
    def structure_dataset(self) -> tuple[PolymlpStructure, np.ndarray]:
        """Return equilibrium structures and positions.

        Return
        ------
        positions_all: shape=(n_samples, 3, n_atom)
        """
        return (self.supercell_polymlp, self._positions_all)

    def supercells(self) -> list[PolymlpStructure]:
        """Return supercells with displacements in PolymlpStructure format."""
        return get_structures_from_multiple_positions(
            self._positions_all,
            self.supercell_polymlp,
        )
