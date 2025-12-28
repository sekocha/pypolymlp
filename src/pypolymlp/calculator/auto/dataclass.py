"""Dataclass for prototype structure."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import get_lattice_constants


@dataclass
class Prototype:
    """Dataclass for prototype structure."""

    structure: PolymlpStructure
    name: str
    icsd_id: int
    n_atom: int
    phonon_supercell: tuple

    structure_eq: Optional[PolymlpStructure] = None
    energy: Optional[float] = None
    elastic_constants: Optional[np.ndarray] = None

    @property
    def lattice_constants(self):
        """Return lattice constants."""
        return get_lattice_constants(self.structure_eq)
