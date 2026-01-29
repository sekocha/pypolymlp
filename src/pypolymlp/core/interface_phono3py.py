"""Interface for phono3py."""

from typing import Optional, Union

import numpy as np
import phono3py
from phono3py.api_phono3py import Phono3py

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.dataset_utils import DatasetDFT
from pypolymlp.core.displacements import get_structures_from_displacements
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure


def parse_phono3py_yaml(
    filename: Union[str, Phono3py],
    element_order: Optional[list[str]] = None,
    use_phonon_dataset: bool = False,
    return_displacements: bool = False,
) -> DatasetDFT:
    """Read phono3py.yaml and return DFT dataclass.

    Parameters
    ----------
    yamlfile: phono3py yaml file
    energies_filename: Energies.
                       The first line must give the energy of perfect supercell

    phono3py.yaml must be composed of datasets for a fixed composition.
    """
    ph3 = Phono3pyYaml(filename, use_phonon_dataset=use_phonon_dataset)

    if ph3.energies is None:
        raise RuntimeError("Energy data not found in phono3py.yaml")

    if not np.all(np.isin(element_order, ph3.supercell_phono3py.symbols)):
        raise ValueError("Elements in the input file are not found in phono3py.yaml.")

    dft = DatasetDFT(
        ph3.supercells,
        ph3.energies,
        forces=ph3.forces,
        stresses=None,
        element_order=element_order,
    )
    if return_displacements:
        return dft, ph3.displacements
    return dft


def parse_structures_from_phono3py_yaml(
    phono3py_yaml: Union[str, Phono3py],
) -> list[PolymlpStructure]:
    """Parse phono3py.yaml and return structures."""
    ph3 = Phono3pyYaml(phono3py_yaml)
    return ph3.supercells


def parse_phono3py_yaml_fcs(
    phono3py_yaml: Union[str, Phono3py],
    use_phonon_dataset: bool = False,
):
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

    def __init__(
        self,
        filename: Union[str, Phono3py],
        use_phonon_dataset: bool = False,
    ):
        """Init method."""
        self._ph3 = self._parse_yaml(filename)
        self._set_properties(use_phonon_dataset=use_phonon_dataset)
        self._supercell = phonopy_cell_to_structure(self._ph3.supercell)
        self._supercells = get_structures_from_displacements(
            self._displacements,
            self._supercell,
        )

    def _parse_yaml(self, filename: Union[str, Phono3py]):
        """Parse phono3py.yaml file."""
        if isinstance(filename, str):
            self._ph3 = phono3py.load(filename, produce_fc=False, log_level=1)
        elif isinstance(filename, Phono3py):
            self._ph3 = filename
        else:
            raise RuntimeError("Filename must be string or Phono3py.")
        return self._ph3

    def _set_properties(self, use_phonon_dataset: bool = False):
        """Set properties."""
        if not use_phonon_dataset:
            self._displacements = self._ph3.displacements.transpose((0, 2, 1))
            self._forces = self._ph3.forces.transpose((0, 2, 1))
            try:
                self._energies = self._ph3.supercell_energies
            except:
                raise RuntimeError("Energy data not found in phono3py.yaml.")
        else:
            print("Using phono3py.phonon_*** dataset", flush=True)
            self._displacements = self._ph3.phonon_dataset["displacements"]
            self._displacements = self._displacements.transpose((0, 2, 1))
            self._forces = self._ph3.phonon_dataset["forces"]
            self._forces = self._forces.transpose((0, 2, 1))
            try:
                self._energies = self._ph3.supercell_energies
            except:
                raise RuntimeError("Energy data not found in phono3py.yaml.")
        return self

    @property
    def phono3py(self):
        """Return phono3py object."""
        return self._ph3

    @property
    def supercell_phono3py(self):
        """Return equilibrium supercell in Phonopy format."""
        return self._ph3.supercell

    @property
    def supercell(self) -> PolymlpStructure:
        """Return equilibrium supercell in PolymlpStructure format."""
        return self._supercell

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
        """Return displacements and forces."""
        return (self.displacements, self.forces)

    @property
    def supercells(self) -> list[PolymlpStructure]:
        """Return supercells with displacements in PolymlpStructure format."""
        return self._supercells
