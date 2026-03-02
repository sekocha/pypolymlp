"""Tests of functions for feature calculations."""

import copy
from pathlib import Path

import numpy as np

from pypolymlp.calculator.compute_formation_energies import (
    PolymlpFormationEnergies,
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


def test_initialize_composition():
    """Test _initialize_composition."""
    end_energies = [0.1, 0.3, 0.5]
    composition = _initialize_composition(end_energies=end_energies)
    np.testing.assert_allclose(composition._proj, np.eye(3))

    composition = _initialize_composition(
        end_structures=end_structures,
        end_energies=end_energies,
        elements=("Cu", "Ag", "Au"),
    )
    np.testing.assert_allclose(composition._composition_axis, np.eye(3) * 4)
    np.testing.assert_allclose(composition._proj, np.eye(3))
    np.testing.assert_allclose(composition.energies_end_members, [0.1, 0.3, 0.5])

    composition = _initialize_composition(
        end_structures=end_structures,
        elements=("Cu", "Ag", "Au"),
        properties=prop,
    )
    np.testing.assert_allclose(composition._composition_axis, np.eye(3) * 4)
    np.testing.assert_allclose(composition._proj, np.eye(3))
    values = [-11.645044684167923, -10.114213158625798, -12.161605091983404]
    np.testing.assert_allclose(composition.energies_end_members, values)


def test_PolymlpFormationEnergies():
    """Test PolymlpFormationEnergies."""
    # structures = parse_structures_from_poscars(poscars)
    # end1 = copy.deepcopy(structures[0])
    # end1.elements = ["Cu" for _ in end1.elements]
    # end1.types = [0 for _ in end1.elements]
    # end2 = structures[0]
    # end3 = copy.deepcopy(structures[0])
    # end3.elements = ["Au" for _ in end1.elements]
    # end3.types = [2 for _ in end1.elements]

    # end_structures = [end1, end2, end3]
    api = PolymlpFormationEnergies(pot=pot)
    api.end_structures = end_structures
    form = api.compute(structures)
    values = [0.0, -1.91101088, -1.91101088, -1.58407348, -1.58407348]
    np.testing.assert_allclose(form, values)
