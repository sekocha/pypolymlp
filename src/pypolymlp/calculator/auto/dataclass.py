"""Dataclass for prototype structure."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import get_lattice_constants
from pypolymlp.utils.yaml_utils import save_cell


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
    volume: Optional[float] = None
    bulk_modulus: Optional[float] = None
    elastic_constants: Optional[np.ndarray] = None

    eos_mlp: Optional[np.ndarray] = None
    eos_fit: Optional[np.ndarray] = None

    @property
    def lattice_constants(self):
        """Return lattice constants."""
        return get_lattice_constants(self.structure_eq)

    def save_properties(self, filename: str = "polymlp_prototype.yaml"):
        """Save properties for prototype."""
        with open(filename, "w") as f:
            save_cell(self.structure_eq, tag="unitcell", file=f)
            print("equilibrium_properties:", file=f)
            print("  energy:      ", self.energy, file=f)
            print("  volume:      ", self.volume, file=f)
            print("  bulk_modulus:", self.bulk_modulus, file=f)
            print(file=f)

            print("eos_data_mlp:", file=f)
            yaml.dump(self.eos_mlp.tolist(), f, default_flow_style=False)
            print(file=f)
            print("eos_data_fit:", file=f)
            yaml.dump(self.eos_fit.tolist(), f, default_flow_style=False)
