"""Tests of utility functions for SSCHA."""

import copy

import numpy as np

from pypolymlp.calculator.sscha.sscha_utils import symmetrize_properties
from pypolymlp.utils.symfc_utils import compute_projector_cartesian
from pypolymlp.utils.tensor_utils import compute_spg_projector_O2


def test_symmetrize_properties(structure_rocksalt):
    """Test symmetrize_properties."""
    st = copy.deepcopy(structure_rocksalt)
    st.positions[0, 0] += 0.01

    proj_f = compute_projector_cartesian(st)
    proj_s = compute_spg_projector_O2(st)

    forces = np.zeros((3, 8))
    forces[0, 0] = 0.1
    stress = np.ones(6)
    stress[0] = 1.2
    stress[1] = 1.1
    forces_sym, stress_sym = symmetrize_properties(
        forces,
        stress,
        proj_f,
        proj_s,
        n_unitcells=1,
    )
    np.testing.assert_allclose(
        forces_sym[0],
        [0.0875, -0.0125, -0.0125, -0.0125, -0.0125, -0.0125, -0.0125, -0.0125],
    )
    np.testing.assert_allclose(
        stress_sym,
        [
            1.2,
            1.05,
            1.05,
            0.0,
            0.0,
            0.0,
        ],
    )
