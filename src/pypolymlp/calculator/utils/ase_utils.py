"""Utility functions for ASE."""

from collections import Counter

import numpy as np
from ase.atoms import Atoms
from phonopy.structure.atoms import PhonopyAtoms, symbol_map

from pypolymlp.core.data_format import PolymlpStructure


def phonopy_atoms_to_ase_atoms(phonopy_atoms: PhonopyAtoms) -> Atoms:
    """Convert PhonopyAtoms to ASE Atoms."""
    return Atoms(
        cell=phonopy_atoms.cell,
        numbers=phonopy_atoms.numbers,
        pbc=True,
        scaled_positions=phonopy_atoms.scaled_positions,
    )


def ase_atoms_to_phonopy_atoms(ase_atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        scaled_positions=ase_atoms.get_scaled_positions(),
        cell=ase_atoms.cell,
    )


def structure_to_ase_atoms(structure: PolymlpStructure) -> Atoms:
    """Convert PolymlpStructure to ASE Atoms."""
    numbers = np.array([symbol_map[sym] for sym in structure.elements], dtype=np.int32)
    return Atoms(
        cell=structure.axis.T,
        scaled_positions=structure.positions.T,
        numbers=numbers,
        pbc=True,
    )


def ase_atoms_to_structure(ase_atoms: Atoms) -> PolymlpStructure:
    """Convert ASE Atoms to PolymlpStructure."""
    elements = ase_atoms.get_chemical_symbols()
    unique_elements = sorted(set(elements), key=elements.index)
    counts = Counter(elements)
    n_atoms = [counts[el] for el in unique_elements]
    types = [i for i, n in enumerate(n_atoms) for _ in range(n)]

    return PolymlpStructure(
        axis=ase_atoms.cell.T,
        positions=ase_atoms.get_scaled_positions().T,
        n_atoms=n_atoms,
        elements=elements,
        types=types,
        volume=ase_atoms.get_volume(),
    )
