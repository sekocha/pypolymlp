"""Tests of polymlp_params.py."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.polymlp_params import (
    set_active_gaussian_params,
    set_all_params,
    set_element_properties,
    set_gaussian_params,
    set_gtinv_params,
    set_regression_alphas,
)

cwd = Path(__file__).parent


def test_set_regression_alphas():
    """Test set_regression_alphas."""
    params = set_regression_alphas((-5, -3, 3))
    np.testing.assert_allclose(params, [-5, -4, -3])


def test_set_element_properties():
    """Test set_element_properties."""
    elements = ["Ag", "Au"]
    elements1, n_type1, atomic_energies1 = set_element_properties(elements)
    assert elements1 == elements
    assert n_type1 == 2
    np.testing.assert_allclose(atomic_energies1, [0.0, 0.0])


def test_set_gtinv_params():
    """Test set_gtinv_params."""
    n_type = 2
    gtinv, max_l = set_gtinv_params(
        n_type, feature_type="gtinv", gtinv_order=3, gtinv_maxl=(2, 2)
    )
    assert max_l == 2
    assert gtinv.order == 3
    assert gtinv.max_l == (2, 2)
    assert len(gtinv.lm_seq) == 9
    assert len(gtinv.l_comb) == 9
    assert len(gtinv.lm_coeffs) == 9
    assert [len(c) for c in gtinv.lm_coeffs] == [1, 1, 3, 5, 1, 3, 5, 9, 19]


def test_set_gaussian_params():
    """Test set_gaussian_params."""
    params = set_gaussian_params(params1=(0.5, 1.0, 2), params2=(0.0, 4.0, 5))
    params_true = [
        [0.5, 0.0],
        [0.5, 1.0],
        [0.5, 2.0],
        [0.5, 3.0],
        [0.5, 4.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
        [0.0, 0.0],
    ]
    np.testing.assert_allclose(params, params_true)


def test_set_active_gaussian_params():
    """Test set_active_gaussian_params."""
    pair_params = set_gaussian_params(params1=(0.5, 1.0, 2), params2=(0.0, 6.0, 10))
    distance = {
        ("Ag", "Ag"): [2.0],
        ("Au", "Au"): [1.0, 2.0],
        ("Au", "Ag"): [1.5, 2.5],
    }
    pair_params_indices, cond = set_active_gaussian_params(
        pair_params,
        elements=("Ag", "Au"),
        distance=distance,
    )
    assert pair_params_indices[(0, 0)] == [1, 2, 3, 4, 5, 12, 13, 14, 20]
    assert pair_params_indices[(0, 1)] == [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 20]
    assert pair_params_indices[(1, 1)] == [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 20]
    assert cond == True

    pair_params_indices, cond = set_active_gaussian_params(
        pair_params,
        elements=("Au", "Ag"),
        distance=distance,
    )
    assert pair_params_indices[(0, 0)] == [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 20]
    assert pair_params_indices[(0, 1)] == [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 20]
    assert pair_params_indices[(1, 1)] == [1, 2, 3, 4, 5, 12, 13, 14, 20]
    assert cond == True

    pair_params_indices, cond = set_active_gaussian_params(
        pair_params,
        elements=("Au", "Ag"),
        distance=None,
    )
    assert pair_params_indices[(0, 0)] == list(range(21))
    assert pair_params_indices[(0, 1)] == list(range(21))
    assert pair_params_indices[(1, 1)] == list(range(21))
    assert cond == False


def test_set_all_params():
    """Test set_all_params."""
    params = set_all_params(
        elements=("Ag", "Au"),
        include_force=True,
        include_stress=False,
        cutoff=6.0,
        model_type=4,
        max_p=2,
        feature_type="gtinv",
        gaussian_params1=(1.0, 1.0, 1),
        gaussian_params2=(0.0, 5.0, 7),
        distance=None,
        reg_alpha_params=(-3.0, 1.0, 5),
        gtinv_order=3,
        gtinv_maxl=(4, 4),
        atomic_energy=None,
    )

    assert params.n_type == 2
    np.testing.assert_equal(params.elements, ["Ag", "Au"])
    model = params.model
    assert model.cutoff == pytest.approx(6.0)
    assert model.model_type == 4
    assert model.max_p == 2
    assert model.max_l == 4
    assert model.feature_type == "gtinv"
    gtinv = params.model.gtinv
    assert gtinv.order == 3
    np.testing.assert_equal(gtinv.max_l, (4, 4))
    assert len(gtinv.lm_seq) == 20
    assert len(gtinv.l_comb) == 20
    assert len(gtinv.lm_coeffs) == 20
    assert model.pair_type == "gaussian"

    np.testing.assert_allclose(
        model.pair_params,
        [
            [1.0, 0.0],
            [1.0, 0.83333333],
            [1.0, 1.66666667],
            [1.0, 2.5],
            [1.0, 3.33333333],
            [1.0, 4.16666667],
            [1.0, 5.0],
            [0.0, 0.0],
        ],
    )
    assert model.pair_params_conditional[(0, 0)] == list(range(8))
    assert model.pair_params_conditional[(0, 1)] == list(range(8))
    assert model.pair_params_conditional[(1, 1)] == list(range(8))
    np.testing.assert_allclose(params.atomic_energy, [0.0, 0.0])
    np.testing.assert_allclose(params.regression_alpha, [-3, -2, -1, 0, 1])
    assert params.include_force == True
    assert params.include_stress == False
    assert params.dataset_type == "vasp"
    np.testing.assert_allclose(params.alphas, [1e-3, 1e-2, 1e-1, 1, 1e1])
