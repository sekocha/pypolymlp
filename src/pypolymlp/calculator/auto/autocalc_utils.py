"""Utility functions for systematic calculations."""

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

    structure_dft: Optional[PolymlpStructure] = None

    eos_mlp: Optional[np.ndarray] = None
    eos_fit: Optional[np.ndarray] = None

    phonon_dos: Optional[np.ndarray] = None
    temperatures: Optional[np.ndarray] = None
    qha_thermal_expansion: Optional[np.ndarray] = None
    qha_bulk_modulus: Optional[np.ndarray] = None

    @property
    def lattice_constants(self):
        """Return lattice constants."""
        if self.structure_eq is None:
            return None
        return get_lattice_constants(self.structure_eq)

    @property
    def lattice_constants_dft(self):
        """Return lattice constants from DFT."""
        if self.structure_dft is None:
            return None
        return get_lattice_constants(self.structure_dft)

    @property
    def volume_dft(self):
        """Return lattice constants from DFT."""
        if self.structure_dft is None:
            return None
        return self.structure_dft.volume

    def set_eos_data(
        self,
        e0: float,
        v0: float,
        b0: float,
        eos_mlp: np.ndarray,
        eos_fit: np.ndarray,
    ):
        """Set EOS properties.

        Parameters must be given using the unit of per cell.
        Properties per atom are assigned to attributes.
        """
        self.energy = e0 / self.n_atom
        self.volume = v0 / self.n_atom
        self.bulk_modulus = b0
        self.eos_mlp = eos_mlp / self.n_atom
        self.eos_fit = eos_fit / self.n_atom
        return self

    def set_qha_data(
        self,
        temperatures: np.ndarray,
        thermal_expansion: np.ndarray,
        bulk_modulus: np.ndarray,
    ):
        """Set QHA properties."""
        self.temperatures = temperatures
        self.qha_thermal_expansion = thermal_expansion
        self.qha_bulk_modulus = bulk_modulus
        return self

    def save_properties(self, filename: str = "polymlp_prototype.yaml"):
        """Save properties for prototype."""
        with open(filename, "w") as f:
            save_cell(self.structure_eq, tag="unitcell", file=f)
            print("structure_type:", self.name, file=f)
            print("icsd_id:       ", self.icsd_id, file=f)
            print(file=f)
            print("equilibrium_properties:", file=f)
            print("  energy:      ", self.energy, file=f)
            print("  volume:      ", self.volume, file=f)
            print("  bulk_modulus:", self.bulk_modulus, file=f)
            print(file=f)

            lattice_constants = self.lattice_constants
            if lattice_constants is not None:
                print("lattice_constants:", file=f)
                a, b, c, calpha, cbeta, cgamma = lattice_constants
                alpha = np.degrees(np.arccos(calpha))
                beta = np.degrees(np.arccos(cbeta))
                gamma = np.degrees(np.arccos(cgamma))
                print("  a:    ", np.round(a, 5), file=f)
                print("  b:    ", np.round(b, 5), file=f)
                print("  c:    ", np.round(c, 5), file=f)
                print("  alpha:", np.round(alpha, 5), file=f)
                print("  beta: ", np.round(beta, 5), file=f)
                print("  gamma:", np.round(gamma, 5), file=f)
                print(file=f)

            lattice_constants_dft = self.lattice_constants_dft
            if lattice_constants_dft is not None:
                print("lattice_constants_dft:", file=f)
                a, b, c, calpha, cbeta, cgamma = lattice_constants_dft
                alpha = np.degrees(np.arccos(calpha))
                beta = np.degrees(np.arccos(cbeta))
                gamma = np.degrees(np.arccos(cgamma))
                print("  a:    ", np.round(a, 5), file=f)
                print("  b:    ", np.round(b, 5), file=f)
                print("  c:    ", np.round(c, 5), file=f)
                print("  alpha:", np.round(alpha, 5), file=f)
                print("  beta: ", np.round(beta, 5), file=f)
                print("  gamma:", np.round(gamma, 5), file=f)
                print(file=f)

                print("volume_dft:", self.volume_dft, file=f)
                print(file=f)

            if self.elastic_constants is not None:
                print("elastic_constants:", file=f)
                elastic_constants = np.round(self.elastic_constants, 2)
                elastic_constants[np.isclose(elastic_constants, 0.0)] = 0.0
                yaml.dump(elastic_constants.tolist(), f, default_flow_style=False)
                print(file=f)

            if self.eos_mlp is not None:
                print("eos_data_mlp:", file=f)
                yaml.dump(self.eos_mlp.tolist(), f, default_flow_style=False)


def get_atomic_size_scales():
    """Return scale."""
    atomic_radius = {
        "H": 0.53,
        "He": 0.31,
        "Li": 1.52,
        "Be": 1.12,
        "B": 0.85,
        "C": 0.70,
        "N": 0.65,
        "O": 0.60,
        "F": 0.50,
        "Ne": 0.38,
        "Na": 1.86,
        "Mg": 1.60,
        "Al": 1.43,
        "Si": 1.17,
        "P": 1.06,
        "S": 1.02,
        "Cl": 0.99,
        "Ar": 0.71,
        "K": 2.03,
        "Ca": 1.74,
        "Sc": 1.44,
        "Ti": 1.32,
        "V": 1.22,
        "Cr": 1.18,
        "Mn": 1.17,
        "Fe": 1.16,
        "Co": 1.11,
        "Ni": 1.10,
        "Cu": 1.28,
        "Zn": 1.33,
        "Ga": 1.26,
        "Ge": 1.22,
        "As": 1.20,
        "Se": 1.16,
        "Br": 1.14,
        "Kr": 1.03,
        "Rb": 2.16,
        "Sr": 1.91,
        "Y": 1.62,
        "Zr": 1.45,
        "Nb": 1.34,
        "Mo": 1.30,
        "Tc": 1.28,
        "Ru": 1.25,
        "Rh": 1.25,
        "Pd": 1.28,
        "Ag": 1.44,
        "Cd": 1.48,
        "In": 1.56,
        "Sn": 1.45,
        "Sb": 1.40,
        "Te": 1.36,
        "I": 1.33,
        "Xe": 1.31,
        "Cs": 2.35,
        "Ba": 1.98,
        "La": 1.87,
        "Ce": 1.82,
        "Pr": 1.82,
        "Nd": 1.82,
        "Pm": 1.82,
        "Sm": 1.80,
        "Eu": 1.85,
        "Gd": 1.80,
        "Tb": 1.79,
        "Dy": 1.78,
        "Ho": 1.78,
        "Er": 1.77,
        "Tm": 1.76,
        "Yb": 1.94,
        "Lu": 1.73,
        "Hf": 1.44,
        "Ta": 1.34,
        "W": 1.30,
        "Re": 1.28,
        "Os": 1.25,
        "Ir": 1.23,
        "Pt": 1.23,
        "Au": 1.44,
        "Hg": 1.49,
        "Tl": 1.56,
        "Pb": 1.46,
        "Bi": 1.48,
        "Po": 1.40,
        "At": 1.50,
        "Rn": 1.50,
        "Fr": 2.60,
        "Ra": 2.21,
    }
    for k, v in atomic_radius.items():
        atomic_radius[k] = v / 1.32
    return atomic_radius
