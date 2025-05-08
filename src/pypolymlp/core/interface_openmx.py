"""Interfaces for openmx files."""

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import BohrtoAng, HartreetoEV


def parse_openmx(filenames: list[str]):
    """Parse openmx log files."""
    energies, forces, structures = [], [], []

    for fname in filenames:
        strs, e, f = parse_single_openmx(fname)
        energies.extend(e)
        forces.extend(f)
        structures.extend(strs)

    return structures, np.array(energies), forces


def parse_single_openmx(filename: str):
    """Parse single openmx log file."""
    with open(filename) as f:
        lines = f.readlines()

    energies, forces, structures = [], [], []
    iline = 0
    while iline < len(lines):
        # parse number of atoms
        natom = int(lines[iline])
        iline += 1
        # parse energy and axis matrix
        sp = lines[iline].split()
        energy = float(sp[4]) * HartreetoEV
        if "Temperature" in lines[iline]:
            axis = np.array([float(s) for s in sp[12:]]).reshape((3, 3)).T
        else:
            axis = np.array([float(s) for s in sp[7:]]).reshape((3, 3)).T
        iline += 1
        # parse positions, elements, and forces
        positions_cartesian, elements, force = [], [], []
        for iatom in range(natom):
            sp = lines[iline].split()
            elements.append(sp[0])
            positions_cartesian.append([float(s) for s in sp[1:4]])
            force.append([float(s) for s in sp[4:7]])
            iline += 1
        positions_cartesian = np.array(positions_cartesian).T
        positions = np.linalg.inv(axis) @ positions_cartesian
        force = np.array(force).T
        force *= HartreetoEV / BohrtoAng
        types, n_atoms = _assign_types(elements)

        energies.append(energy)
        forces.append(force)
        st = PolymlpStructure(
            axis=axis,
            positions=positions,
            n_atoms=n_atoms,
            elements=elements,
            types=types,
        )
        structures.append(st)

    return structures, energies, forces


def _assign_types(elements: list):
    """Assign types from element list."""
    uniq_elements = np.unique(elements)
    types = np.zeros(len(elements), dtype=int)
    n_atoms = []
    for i, ele in enumerate(uniq_elements):
        match = np.array(elements) == ele
        types[match] = i
        n_atoms.append(np.count_nonzero(match))

    return types, n_atoms
