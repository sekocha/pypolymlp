"""Utility functions for phonopy."""

import io
from collections import Counter
from typing import Optional

import numpy as np
import yaml
from phonopy import Phonopy
from phonopy.interface.vasp import VasprunxmlExpat
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import Bohr, Hartree

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure


def structure_to_phonopy_cell(structure: PolymlpStructure) -> PhonopyAtoms:
    """Convert PolymlpStructure to phonopyAtoms."""
    ph_cell = PhonopyAtoms(
        symbols=structure.elements,
        cell=structure.axis.T,
        scaled_positions=structure.positions.T,
    )
    return ph_cell


def phonopy_cell_to_structure(ph_cell: PhonopyAtoms) -> PolymlpStructure:
    """Convert PhonopyAtoms to PolymlpStructure."""
    elements = ph_cell.symbols
    elements_uniq = sorted(set(elements), key=elements.index)
    elements_count = Counter(elements)
    n_atoms = [elements_count[ele] for ele in elements_uniq]
    types = [i for i, n in enumerate(n_atoms) for _ in range(n)]

    structure = PolymlpStructure(
        axis=ph_cell.cell.T,
        positions=ph_cell.scaled_positions.T,
        n_atoms=n_atoms,
        elements=elements,
        types=types,
        volume=ph_cell.volume,
    )
    return structure


def phonopy_supercell(
    structure: PolymlpStructure,
    supercell_matrix: np.ndarray = None,
    supercell_diag: np.ndarray = None,
    return_phonopy: bool = True,
) -> PolymlpStructure:
    """Generate supercell in Phonopy format."""
    if supercell_diag is not None:
        supercell_matrix = np.diag(supercell_diag)

    unitcell = structure_to_phonopy_cell(structure)
    supercell = Phonopy(unitcell, supercell_matrix).supercell
    if return_phonopy:
        return supercell

    supercell = phonopy_cell_to_structure(supercell)
    supercell.supercell_matrix = supercell_matrix
    return supercell


def compute_forces_phonopy_displacements(
    ph: Phonopy,
    pot: str = "polymlp.yaml",
    distance: float = 0.01,
) -> np.ndarray:
    """Compute forces using phonopy object and polymlp.

    Return
    ------
    forces: (n_str, n_atom, 3)
    """
    prop = Properties(pot=pot)

    ph.generate_displacements(distance=distance)
    supercells = ph.supercells_with_displacements
    structures = [phonopy_cell_to_structure(cell) for cell in supercells]
    """ forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)"""
    _, forces, _ = prop.eval_multiple(structures)
    forces = np.array(forces).transpose((0, 2, 1))
    return forces


def get_nac_params(
    vasprun: Optional[str] = None,
    supercell_matrix: Optional[np.ndarray] = None,
):
    """Get NAC parameters."""
    if vasprun is None:
        return None

    with io.open(vasprun, "rb") as f:
        vasprun = VasprunxmlExpat(f)
        vasprun.parse()
        epsilon = vasprun.epsilon
        borns = vasprun.born
        unitcell = vasprun.cell

    borns_, epsilon_ = symmetrize_borns_and_epsilon(
        borns,
        epsilon,
        unitcell,
        supercell_matrix=supercell_matrix,
        symprec=1e-5,
    )
    nac_params = {
        "born": borns_,
        "factor": Hartree * Bohr,
        "dielectric": epsilon_,
    }
    return nac_params


def get_frequency(yamlname):
    """Return phonon frequency from phonopy yaml."""
    data = yaml.load(open(yamlname))
    freq = [b["frequency"] for d in data["phonon"] for b in d["band"]]
    weight = [d["weight"] for d in data["phonon"] for b in d["band"]]
    return np.array(freq), np.array(weight)


def get_band_frequency(yamlname):
    """Return band frequency from phonopy yaml."""
    data = yaml.load(open(yamlname))
    dist = [d["distance"] for d in data["phonon"]]
    freq = [[b["frequency"] for b in d["band"]] for d in data["phonon"]]
    return dist, freq
