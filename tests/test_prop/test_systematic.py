"""Tests of systematic property calculations."""

import numpy as np
import pytest
from data_MgO import params as params_MgO
from data_Ti import params as params_Ti

from pypolymlp.calculator.utils.properties_utils import convert_stresses_in_gpa
from pypolymlp.calculator.validation.validation import PolymlpValidation


def check_eval(unitcell, api, te, tf, ts):
    """Test properties."""
    etol = 1e-8
    ftol = 1e-7
    stol = 1e-6

    e, f, s = api.eval(unitcell)
    assert e == pytest.approx(te, abs=etol)
    np.testing.assert_allclose(f.T, tf, atol=ftol)
    s = convert_stresses_in_gpa(np.array([s]), [unitcell])[0]
    np.testing.assert_allclose(s, ts, atol=stol)


params = []
params.extend(params_Ti)
params.extend(params_MgO)


@pytest.mark.parametrize("unitcell, api, te, tf, ts", params)
@pytest.mark.parametrize(
    "check_method",
    [
        "check_eval",
        "check_rotation",
        "check_translation",
        "check_permute_axis",
        "check_permute_atom",
        "check_different_types_representation",
        "check_different_primitive",
        "check_supercell",
    ],
)
def test_eval_and_invariance(unitcell, api, te, tf, ts, check_method):
    """Test evaluations and invariance."""

    if check_method == "check_eval":
        check_eval(unitcell, api, te, tf, ts)
        return

    val = PolymlpValidation(unitcell, api)
    try:
        getattr(val, check_method)(
            etol=1e-12,
            ftol=1e-12,
            stol=1e-12,
        )
    except:
        getattr(val, check_method)(
            etol=1e-12,
        )
