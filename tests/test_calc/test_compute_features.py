"""Tests of functions for feature calculations."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.compute_features import (
    compute_from_infile,
    compute_from_polymlp,
    update_types,
)
from pypolymlp.core.interface_vasp import parse_structures_from_poscars

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscars = [
    path_file + "poscars/POSCAR-00001.MgO",
    path_file + "poscars/POSCAR-00002.MgO",
]

# pot = path_file + "mlps/polymlp.yaml.pair.MgO"
# pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
# infile = path_file + "mlps/polymlp.in.gtinv.MgO"


def test_update_types():
    """Test update_types."""
    structures = parse_structures_from_poscars(poscars)
    element_order = ["Mg", "O"]
    updated_structures = update_types(structures, element_order)
    st = updated_structures[0]
    assert st.types[0] == 0
    assert st.types[32] == 1
    assert st.elements[0] == "Mg"
    assert st.elements[32] == "O"

    element_order = ["O", "Mg"]
    updated_structures = update_types(structures, element_order)
    st = updated_structures[0]
    assert st.types[0] == 0
    assert st.types[32] == 1
    assert st.elements[0] == "O"
    assert st.elements[32] == "Mg"


def test_features_from_polymlp1():
    """Test feature calculations from polymlp."""
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"
    structures = parse_structures_from_poscars(poscars)
    x = compute_from_polymlp(structures, pot=pot, return_mlp_dict=False)
    assert x.shape == (2, 324)
    assert np.sum(x) == pytest.approx(997193.0146734761, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-6.428600229143143, rel=1e-6)

    x, _ = compute_from_polymlp(structures, pot=pot, force=True, return_mlp_dict=True)
    assert x.shape == (398, 324)
    x = compute_from_polymlp(
        structures, pot=pot, force=True, stress=True, return_mlp_dict=False
    )
    assert x.shape == (398, 324)


def test_features_from_polymlp2():
    """Test feature calculations from polymlp."""
    pot = path_file + "mlps/polymlp.yaml.gtinv.MgO"
    structures = parse_structures_from_poscars(poscars)
    x = compute_from_polymlp(structures, pot=pot, return_mlp_dict=False)
    assert x.shape == (2, 1899)
    assert np.sum(x) == pytest.approx(237100.3979091199, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-1.9717588279337908, rel=1e-6)


def test_features_from_infile1():
    """Test feature calculations from input file."""
    infile = path_file + "mlps/polymlp.in.gtinv.MgO"
    structures = parse_structures_from_poscars(poscars)
    x = compute_from_infile(infile, structures, force=False)
    assert x.shape == (2, 1899)
    assert np.sum(x) == pytest.approx(237100.3979091199, rel=1e-6)
    assert np.sum(x[0] - x[1]) == pytest.approx(-1.9717588279337908, rel=1e-6)

    x = compute_from_infile(infile, structures)
    assert x.shape == (398, 1899)
