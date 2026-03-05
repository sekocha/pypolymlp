"""Tests of functions for feature calculations."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.calculator.compute_formation_energies import (
    PolymlpFormationEnergies,
    _get_n_atoms,
    _initialize_composition,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import parse_structures_from_poscars

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "mlps/polymlp.lammps.gtinv.Cu-Ag-Au"
prop = Properties(pot=pot)

poscars = [
    path_file + "poscars/POSCAR1.Cu-Ag-Au",
    path_file + "poscars/POSCAR2.Cu-Ag-Au",
    path_file + "poscars/POSCAR3.Cu-Ag-Au",
    path_file + "poscars/POSCAR4.Cu-Ag-Au",
    path_file + "poscars/POSCAR5.Cu-Ag-Au",
]
structures = parse_structures_from_poscars(poscars)

end1 = copy.deepcopy(structures[0])
end1.elements = ["Cu" for _ in end1.elements]
end1.types = [0 for _ in end1.elements]
end2 = structures[0]
end3 = copy.deepcopy(structures[0])
end3.elements = ["Au" for _ in end1.elements]
end3.types = [2 for _ in end1.elements]

end_structures = [end1, end2, end3]
elements = ("Cu", "Ag", "Au")
end_energies = np.array([-11.64504468, -10.11421316, -12.16160509])


def test_initialize_composition():
    """Test _initialize_composition."""
    composition = _initialize_composition(end_energies=end_energies)
    np.testing.assert_allclose(composition._proj, np.eye(3))

    composition = _initialize_composition(
        end_structures=end_structures,
        end_energies=end_energies,
        elements=elements,
    )
    np.testing.assert_allclose(composition._composition_axis, np.eye(3) * 4)
    np.testing.assert_allclose(composition._proj, np.eye(3))
    np.testing.assert_allclose(composition.energies_end_members, end_energies)

    composition = _initialize_composition(
        end_structures=end_structures,
        elements=elements,
        properties=prop,
    )
    np.testing.assert_allclose(composition._composition_axis, np.eye(3) * 4)
    np.testing.assert_allclose(composition._proj, np.eye(3))
    np.testing.assert_allclose(composition.energies_end_members, end_energies)


def test_get_n_atoms():
    """Test get_n_atoms."""
    n_atoms_true = np.array(
        [[0, 4, 0], [36, 0, 12], [36, 0, 12], [36, 8, 4], [36, 8, 4]]
    )

    n_atoms = _get_n_atoms(structures, elements=("Cu", "Ag", "Au"))
    np.testing.assert_equal(n_atoms, n_atoms_true)

    n_atoms = _get_n_atoms(structures, elements=("Ag", "Cu", "Au"))
    np.testing.assert_equal(n_atoms, n_atoms_true[:, [1, 0, 2]])

    n_atoms = _get_n_atoms(structures, elements=("Au", "Cu", "Ag"))
    np.testing.assert_equal(n_atoms, n_atoms_true[:, [2, 0, 1]])


def test_PolymlpFormationEnergies():
    """Test PolymlpFormationEnergies."""
    api = PolymlpFormationEnergies(properties=prop)

    values = np.array([0.0, -1.91101088, -1.91101088, -1.58407348, -1.58407348])
    compositions = [
        [0.0, 1.0, 0.0],
        [0.75, 0.0, 0.25],
        [0.75, 0.0, 0.25],
        [0.75, 0.16666667, 0.08333333],
        [0.75, 0.16666667, 0.08333333],
    ]
    end_energies = np.array([-11.64504468, -10.11421316, -12.16160509])

    api.define_end_members(structures=end_structures)
    data = api.compute(structures)
    np.testing.assert_allclose(data[:, -1], values, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)

    api.define_end_members(structures=end_structures, energies=end_energies)
    data = api.compute(structures)
    np.testing.assert_allclose(data[:, -1], values, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)

    api.define_end_members(energies=end_energies / 4)
    data = api.compute(structures)
    np.testing.assert_allclose(data[:, -1], values / 4, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)


def test_PolymlpFormationEnergies2():
    """Test PolymlpFormationEnergies without using MLP calculations."""
    api = PolymlpFormationEnergies(elements=elements)

    values = np.array([0.0, -1.91101088, -1.91101088, -1.58407348, -1.58407348])
    compositions = [
        [0.0, 1.0, 0.0],
        [0.75, 0.0, 0.25],
        [0.75, 0.0, 0.25],
        [0.75, 0.16666667, 0.08333333],
        [0.75, 0.16666667, 0.08333333],
    ]
    end_energies = np.array([-11.64504468, -10.11421316, -12.16160509])

    energies = [
        -10.11421316,
        -164.22234805,
        -164.22234805,
        -156.20431534,
        -156.20431534,
    ]
    api.define_end_members(energies=end_energies / 4)
    data = api.compute(structures, energies=energies)
    np.testing.assert_allclose(data[:, -1], values / 4, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)


def test_PolymlpFormationEnergies3():
    """Test PolymlpFormationEnergies for structures with element swappings."""
    values = np.array([0.75, 0.16666667, 0.08333333, -1.58407348])

    api = PolymlpFormationEnergies(properties=prop)
    api.define_end_members(structures=end_structures, energies=end_energies)

    data = api.compute(structures[-1:])
    np.testing.assert_allclose(data[0], values, atol=1e-7)

    api = PolymlpFormationEnergies(properties=prop)
    api._elements = ("Ag", "Au", "Cu")
    api.define_end_members(structures=end_structures, energies=end_energies)
    data = api.compute(structures[-1:])
    np.testing.assert_allclose(data[0], values, atol=1e-7)

    api = PolymlpFormationEnergies(properties=prop)
    order = [0, 2, 1]
    api.define_end_members(
        structures=[end_structures[i] for i in order],
        energies=[end_energies[i] for i in order],
    )
    data = api.compute(structures[-1:])
    np.testing.assert_allclose(data[0], values[[0, 2, 1, 3]], atol=1e-7)


def test_PolymlpFormationEnergies_convex():
    """Test PolymlpFormationEnergies to obtain convex hull."""

    api = PolymlpFormationEnergies(elements=("Cu", "Ag"))
    api.define_end_members(energies=[0.1, 0.2])
    api._data = np.array(
        [
            [0.8, 0.2, -0.03495486],
            [0.5, 0.5, -0.04193504],
            [0.75, 0.25, -0.04421533],
            [0.75, 0.25, -0.04305931],
            [0.66666667, 0.33333333, -0.00756026],
            [0.16666667, 0.83333333, -0.02537713],
            [0.6, 0.4, -0.02018703],
            [0.75, 0.25, -0.04391786],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    api._structure_names = [str(i) for i in range(api._data.shape[0])]

    convex = api.convex_hull()
    assert convex.shape == (5, 3)
    assert api.structure_names_convex == ["8", "2", "1", "5", "9"]
    assert api.has_end_members

    api._data = np.array(
        [
            [0.8, 0.2, -0.03495486],
            [0.5, 0.5, -0.04193504],
            [0.75, 0.25, -0.04421533],
            [0.75, 0.25, -0.04305931],
            [0.66666667, 0.33333333, -0.00756026],
            [0.16666667, 0.83333333, -0.02537713],
            [0.6, 0.4, -0.02018703],
            [0.75, 0.25, -0.04391786],
        ]
    )
    convex = api.convex_hull()
    assert convex.shape == (5, 3)
