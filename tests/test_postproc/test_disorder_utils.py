"""Tests disorder_utils."""

import copy
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.params import PolymlpParams
from pypolymlp.postproc.disorder_utils import (
    _generate_substitutional_indices,
    _reorder,
    eval_substitutional_structures,
    generate_substitutional_structures,
    set_full_occupancy,
)

cwd = Path(__file__).parent


def test_check_occupancy1():
    """Test check_occupancy."""
    single = PolymlpParamsSingle(
        n_type=3,
        elements=("Cu", "Ag", "Au"),
        model=None,
    )
    params = PolymlpParams(single)

    occ = [[("Cu", 0.25), ("Ag", 0.5), ("Au", 0.25)]]
    occupancy = set_full_occupancy(params, occ)

    assert occupancy[0][0] == ("Cu", 0, 0.25)
    assert occupancy[0][1] == ("Ag", 1, 0.5)
    assert occupancy[0][2] == ("Au", 2, 0.25)


def test_check_occupancy2():
    """Test check_occupancy."""
    single = PolymlpParamsSingle(
        n_type=3,
        elements=("Sr", "Cu", "O"),
        model=None,
    )
    params = PolymlpParams(single)

    occ = [[("Sr", 2 / 3), ("Cu", 1 / 3)], [("Cu", 2 / 3), ("Sr", 1 / 3)], [("O", 1.0)]]
    occupancy = set_full_occupancy(params, occ)

    assert occupancy[0][0][0] == "Sr"
    assert occupancy[0][0][1] == 0
    assert occupancy[0][0][2] == pytest.approx(2 / 3)

    assert occupancy[0][1][0] == "Cu"
    assert occupancy[0][1][1] == 1
    assert occupancy[0][1][2] == pytest.approx(1 / 3)

    assert occupancy[1][0][0] == "Cu"
    assert occupancy[1][0][1] == 1
    assert occupancy[1][0][2] == pytest.approx(2 / 3)

    assert occupancy[1][1][0] == "Sr"
    assert occupancy[1][1][1] == 0
    assert occupancy[1][1][2] == pytest.approx(1 / 3)

    assert occupancy[2][0][0] == "O"
    assert occupancy[2][0][1] == 2
    assert occupancy[2][0][2] == pytest.approx(1.0)


def test_check_occupancy3():
    """Test check_occupancy."""
    single = PolymlpParamsSingle(
        n_type=4,
        elements=("La", "Cu", "Te", "O"),
        model=None,
    )
    params = PolymlpParams(single)

    occ = [
        [("Te", 1.0)],
        [("La", 5 / 6), ("Cu", 1 / 6)],
        [("Cu", 2 / 3), ("La", 1 / 3)],
        [("O", 1.0)],
    ]
    occupancy = set_full_occupancy(params, occ)

    assert occupancy[0][0][0] == "Te"
    assert occupancy[0][0][1] == 2
    assert occupancy[0][0][2] == pytest.approx(1.0)

    assert occupancy[1][0][0] == "La"
    assert occupancy[1][0][1] == 0
    assert occupancy[1][0][2] == pytest.approx(5 / 6)

    assert occupancy[1][1][0] == "Cu"
    assert occupancy[1][1][1] == 1
    assert occupancy[1][1][2] == pytest.approx(1 / 6)

    assert occupancy[2][0][0] == "Cu"
    assert occupancy[2][0][1] == 1
    assert occupancy[2][0][2] == pytest.approx(2 / 3)

    assert occupancy[2][1][0] == "La"
    assert occupancy[2][1][1] == 0
    assert occupancy[2][1][2] == pytest.approx(1 / 3)

    assert occupancy[3][0][0] == "O"
    assert occupancy[3][0][1] == 3
    assert occupancy[3][0][2] == pytest.approx(1.0)


def test_check_occupancy_with_spin():
    """Test check_occupancy."""
    single = PolymlpParamsSingle(
        n_type=2,
        elements=("Fe", "C"),
        model=None,
        atomic_energy=(0.0, 0.0),
        enable_spins=(True, False),
    )
    params = PolymlpParams(single)
    occ = [[("C", 1.0)], [(("Fe", 0), 1 / 2), (("Fe", 1), 1 / 2)]]
    occupancy = set_full_occupancy(params, occ)

    assert occupancy[0][0][0] == "C"
    assert occupancy[0][0][1] == 2
    assert occupancy[0][0][2] == pytest.approx(1.0)

    assert occupancy[1][0][0] == "Fe"
    assert occupancy[1][0][1] == 0
    assert occupancy[1][0][2] == pytest.approx(1 / 2)

    assert occupancy[1][1][0] == "Fe"
    assert occupancy[1][1][1] == 1
    assert occupancy[1][1][2] == pytest.approx(1 / 2)


def test_generate_substitutional_structures(structure_rocksalt):
    """Test generate_substitutional_structures."""
    elements = ("Mg", "Na", "O", "Cl")
    single = PolymlpParamsSingle(n_type=4, elements=elements, model=None)
    params = PolymlpParams(single)

    occ_input = [[("Mg", 0.5), ("Na", 0.5)], [("O", 0.5), ("Cl", 0.5)]]
    occ = set_full_occupancy(params, occ_input)

    ids = _generate_substitutional_indices(structure_rocksalt, occ)
    assert len(ids[("Mg", 0)]) == 2
    assert len(ids[("Na", 1)]) == 2
    assert len(ids[("O", 2)]) == 2
    assert len(ids[("Cl", 3)]) == 2

    structures, orders = generate_substitutional_structures(
        structure_rocksalt,
        occ,
        n_samples=5,
    )
    assert len(structures) == 5
    assert orders.shape == (5, 8)


def test_eval_substitutional_structures(structure_rocksalt):
    """Test eval_substitutional_structures."""
    calc = PypolymlpCalc(pot=str(cwd) + "/../files/polymlp.yaml.MgO")
    lattice = copy.deepcopy(structure_rocksalt)
    lattice.n_atoms = [8]
    lattice.types = np.ones(8, dtype=int)
    lattice.elements = ["Mg"] * 8
    lattice.positions[0, 0] = 0.0001

    elements = ("Mg", "O")
    single = PolymlpParamsSingle(n_type=2, elements=elements, model=None)
    params = PolymlpParams(single)
    occ_input = [[("Mg", 0.5), ("O", 0.5)]]
    occ = set_full_occupancy(params, occ_input)

    e, f, s = eval_substitutional_structures(calc, lattice, occ, n_samples=5)

    assert e.shape == (5,)
    assert np.array(f).shape == (5, 3, 8)
    assert s.shape == (5, 6)

    structures, orders = generate_substitutional_structures(lattice, occ, n_samples=1)
    positions = _reorder(structures[0].positions, orders[0])
    np.testing.assert_allclose(lattice.positions, positions)
