"""Tests disorder_utils."""

from pathlib import Path

from pypolymlp.core.data_format import PolymlpParamsSingle
from pypolymlp.core.params import PolymlpParams
from pypolymlp.postproc.disorder import check_occupancy

cwd = Path(__file__).parent


def test_check_occupancy():
    """Test check_occupancy."""
    single = PolymlpParamsSingle(
        n_type=3,
        elements=("Cu", "Ag", "Au"),
        model=None,
    )
    params = PolymlpParams(single)
    print(params)
    print(params.enable_spins)

    occ = [[("Cu", 0.25), ("Ag", 0.5), ("Au", 0.25)]]
    map_element_to_type = check_occupancy(params, occ)
    print(map_element_to_type)
    assert 1 == 0
