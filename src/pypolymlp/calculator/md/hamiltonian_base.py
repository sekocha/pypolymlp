"""Base class for defining Hamiltonian."""

from abc import ABC, abstractmethod
from typing import Optional

from pypolymlp.core.data_format import PolymlpStructure


class HamiltonianBase(ABC):
    """Base class for defining Hamiltonian."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

    @abstractmethod
    def eval(self, structure: PolymlpStructure, args: Optional[dict] = None):
        """Evaluate energy, forces, and stress tensor.

        Parameters
        ----------
        structure: Structure in PolymlpStructure format.
        args: Parameters used to evaluate properties.
        """
        pass

    @abstractmethod
    def energy(self):
        """Return energy.

        Return
        ------
        energy: unit: eV/supercell.
        """
        pass

    @abstractmethod
    def forces(self):
        """Return forces.

        Return
        ------
        forces: unit: eV/angstrom (3, n_atom)
        """
        pass

    @abstractmethod
    def stress_tensor(self):
        """Return stress tensor.

        Return
        ------
        stress: unit: eV/supercell (6) in the order of xx, yy, zz, xy, yz, zx.
        """
        pass
