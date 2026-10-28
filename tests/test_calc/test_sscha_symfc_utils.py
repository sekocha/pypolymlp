"""Tests of symfc_utils_O2."""

import numpy as np
import pytest

from pypolymlp.calculator.sscha.sscha_core import SSCHACore
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.sscha.symfc_utils_O2 import project_fc2_space, recover_fc2


def test_symfc_utils(unitcell_mlp_Al):
    """Test SSCHACore."""
    unitcell, pot, prop = unitcell_mlp_Al
    size = (2, 2, 2)
    sscha_params = SSCHAParams(unitcell, size, pot=pot, path="tmp")
    sscha = SSCHACore(sscha_params, prop, verbose=True)
    symfc = sscha._symfc

    fc2 = recover_fc2(symfc, np.ones(11))
    assert fc2.shape == (32, 32, 3, 3)
    assert fc2[0, 0, 0, 0] == pytest.approx(-0.004642281307253422)
    assert np.average(fc2) == pytest.approx(0.0)

    fc2 = project_fc2_space(symfc, np.ones((32, 32, 3, 3)))
    assert fc2.shape == (32, 32, 3, 3)
    assert fc2[0, 0, 0, 0] == pytest.approx(0.0)
    assert np.average(fc2) == pytest.approx(0.0)
