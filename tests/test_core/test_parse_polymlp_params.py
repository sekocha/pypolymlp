"""Tests of parse_polymlp_params.py."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.dataset import Dataset
from pypolymlp.core.parser_polymlp_params import parse_parameter_files

cwd = Path(__file__).parent


def test_parse_parameter_files():
    """Test parse_parameter_files."""
    params, common_params, hybrid_params = parse_parameter_files(
        str(cwd) + "/../files/polymlp.in"
    )
    assert params == common_params
    assert hybrid_params is None

    assert params.n_type == 2
    np.testing.assert_equal(common_params.elements, ["Mg", "O"])
    model = params.model
    assert model.cutoff == pytest.approx(8.0)
    assert model.model_type == 3
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
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 6.0],
            [1.0, 7.0],
            [0.0, 0.0],
        ],
    )
    assert model.pair_params_conditional[(0, 0)] == list(range(9))
    assert model.pair_params_conditional[(0, 1)] == list(range(9))
    assert model.pair_params_conditional[(1, 1)] == list(range(9))
    np.testing.assert_allclose(params.atomic_energy, [-0.00040000, -1.85321219])
    np.testing.assert_allclose(params.regression_alpha, [-3, -2, -1, 0, 1])
    assert params.include_force == True
    assert params.include_stress == True
    assert params.dataset_type == "vasp"
    np.testing.assert_allclose(params.alphas, [1e-3, 1e-2, 1e-1, 1, 1e1])

    dft_train_true = [
        Dataset(
            dataset_type="vasp",
            string_list=["dataset/vasprun-*.xml.polymlp"],
            location="dataset/vasprun-*.xml.polymlp",
            files=[],
            include_force=True,
            weight=1.0,
            name="dataset/vasprun-*.xml.polymlp",
            split=True,
        )
    ]
    dft_test_true = [
        Dataset(
            dataset_type="vasp",
            string_list=["dataset/vasprun-*.xml.polymlp"],
            location="dataset/vasprun-*.xml.polymlp",
        )
    ]
    assert params.dft_train == dft_train_true
    assert params.dft_test == dft_test_true


def test_parse_parameter_files_hybrid():
    """Test parse_parameter_files from hybrid files."""
    params, common_params, hybrid_params = parse_parameter_files(
        [str(cwd) + "/../files/polymlp.in", str(cwd) + "/../files/polymlp.in.2"]
    )
    assert params == hybrid_params
    assert len(hybrid_params) == 2

    assert common_params.n_type == 2
    assert common_params.atomic_energy == (-0.0004, -1.85321219)
    np.testing.assert_equal(common_params.elements, ["Mg", "O"])
    np.testing.assert_equal(common_params.element_order, ["Mg", "O"])
