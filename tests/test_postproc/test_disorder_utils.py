"""Tests disorder_utils."""

from pathlib import Path

from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.params import PolymlpParams
from pypolymlp.postproc.disorder_utils import set_element_map

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
    map_element_to_type = set_element_map(params, occ)
    assert map_element_to_type["Cu"] == 0
    assert map_element_to_type["Ag"] == 1
    assert map_element_to_type["Au"] == 2


def test_check_occupancy2():
    """Test check_occupancy."""
    single = PolymlpParamsSingle(
        n_type=3,
        elements=("Sr", "Cu", "O"),
        model=None,
    )
    params = PolymlpParams(single)

    occ = [[("Sr", 2 / 3), ("Cu", 1 / 3)], [("Cu", 2 / 3), ("Sr", 1 / 3)], [("O", 1.0)]]
    map_element_to_type = set_element_map(params, occ)

    assert map_element_to_type["Sr"] == 0
    assert map_element_to_type["Cu"] == 1
    assert map_element_to_type["O"] == 2


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
    map_element_to_type = set_element_map(params, occ)

    assert map_element_to_type["La"] == 0
    assert map_element_to_type["Cu"] == 1
    assert map_element_to_type["Te"] == 2
    assert map_element_to_type["O"] == 3


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
    map_element_to_type = set_element_map(params, occ)

    assert map_element_to_type[("Fe", 0)] == 0
    assert map_element_to_type[("Fe", 1)] == 1
    assert map_element_to_type["C"] == 2
