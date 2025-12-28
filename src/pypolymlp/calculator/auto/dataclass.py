"""Dataclass for prototype structure."""

from dataclasses import dataclass

from pypolymlp.core.data_format import PolymlpStructure


@dataclass
class Prototype:
    """Dataclass for prototype structure."""

    structure: PolymlpStructure
    name: str
    icsd_id: int
    n_atom: int
    phonon_supercell: tuple
