"""Tests of API for calculating formation energies."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
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


def test_api_formation_energy1():
    """Test run_formation_energy in PypolymlpCalc."""
    api = PypolymlpCalc(properties=prop)

    values = np.array([0.0, -1.91101088, -1.91101088, -1.58407348, -1.58407348])
    compositions = [
        [0.0, 1.0, 0.0],
        [0.75, 0.0, 0.25],
        [0.75, 0.0, 0.25],
        [0.75, 0.16666667, 0.08333333],
        [0.75, 0.16666667, 0.08333333],
    ]

    api.init_formation_energy(end_structures=end_structures)
    data = api.run_formation_energy(structures)
    np.testing.assert_allclose(data[:, -1], values, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)

    api.init_formation_energy(end_structures=end_structures, end_energies=end_energies)
    data = api.run_formation_energy(structures)
    np.testing.assert_allclose(data[:, -1], values, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)

    api.init_formation_energy(end_energies=end_energies / 4)
    api.structures = structures
    data = api.run_formation_energy()
    np.testing.assert_allclose(data[:, -1], values / 4, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)


def test_convex():
    """Test convex hull calculations in PypolymlpCalc."""
    values = np.array([0.0, -1.91101088, -1.91101088, -1.58407348, -1.58407348])
    compositions = [
        [0.0, 1.0, 0.0],
        [0.75, 0.0, 0.25],
        [0.75, 0.0, 0.25],
        [0.75, 0.16666667, 0.08333333],
        [0.75, 0.16666667, 0.08333333],
    ]

    api = PypolymlpCalc(properties=prop)
    api.init_formation_energy(end_structures=end_structures, end_energies=end_energies)
    data, convex, convex_names = api.run_formation_energy(structures, convex_hull=True)
    np.testing.assert_allclose(data[:, -1], values, atol=1e-7)
    np.testing.assert_allclose(data[:, :-1], compositions, atol=1e-7)
    assert convex.shape == (5, 4)
    assert len(convex_names) == 5
