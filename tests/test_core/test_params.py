"""Tests of params.py."""

from pathlib import Path

from pypolymlp.core.params import PolymlpParamsList

# import numpy as np
# import pytest


cwd = Path(__file__).parent


def test_PolymlpParamsList1(regdata_mp_149):
    """Test PolymlpParamsList."""
    params, _ = regdata_mp_149
    params_list = PolymlpParamsList(params)
    params_list.append(params)
    assert len(params_list) == 2
    for params in params_list:
        assert params.n_type == 1
        assert params.include_force
