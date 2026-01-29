"""Tests of parser_polymlp_params.py."""

from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.parser_polymlp_params import ParamsParser

cwd = Path(__file__).parent


def test_parse_parameter_files():
    """Test parse_parameter_files."""
    parser = ParamsParser(str(cwd) + "/../files/polymlp.in")
    params = parser.params
    common_params = parser.common_params
    hybrid_params = parser.hybrid_params

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

    train = parser.train[0]
    assert train.files == []
    assert train.include_force == True
    assert train.weight == pytest.approx(1.0)
    assert train.name == "Train_dataset/vasprun-*.xml.polymlp"

    test = parser.test[0]
    assert test.files == []
    assert test.include_force == True
    assert test.weight == pytest.approx(1.0)
    assert test.name == "Test_dataset/vasprun-*.xml.polymlp"


def test_parse_parameter_files_hybrid():
    """Test parse_parameter_files from hybrid files."""
    parser = ParamsParser(
        [str(cwd) + "/../files/polymlp.in", str(cwd) + "/../files/polymlp.in.2"]
    )
    params = parser.params
    common_params = parser.common_params
    hybrid_params = parser.hybrid_params

    assert params == hybrid_params
    assert len(hybrid_params) == 2

    assert common_params.n_type == 2
    assert common_params.atomic_energy == (-0.0004, -1.85321219)
    np.testing.assert_equal(common_params.elements, ["Mg", "O"])
    np.testing.assert_equal(common_params.element_order, ["Mg", "O"])

    train = parser.train[0]
    assert train.files == []
    assert train.include_force == True
    assert train.weight == pytest.approx(1.0)
    assert train.name == "Train_dataset/vasprun-*.xml.polymlp"

    test = parser.test[0]
    assert test.files == []
    assert test.include_force == True
    assert test.weight == pytest.approx(1.0)
    assert test.name == "Test_dataset/vasprun-*.xml.polymlp"


def test_parse_parameter_files_hybrid2():
    """Test parse_parameter_files from hybrid files."""
    parser = ParamsParser(
        [str(cwd) + "/../files/polymlp.in", str(cwd) + "/../files/polymlp.in.2"],
        parse_dft=False,
    )
    params = parser.params
    common_params = parser.common_params
    hybrid_params = parser.hybrid_params

    assert params == hybrid_params
    assert len(hybrid_params) == 2

    assert common_params.n_type == 2
    assert common_params.atomic_energy == (-0.0004, -1.85321219)
    np.testing.assert_equal(common_params.elements, ["Mg", "O"])
    np.testing.assert_equal(common_params.element_order, ["Mg", "O"])


def test_parse_parameter_files2():
    """Test parse_parameter_files."""
    parser = ParamsParser(str(cwd) + "/../files/polymlp.in.infiletest")
    params = parser.params

    model = params.model
    assert model.pair_params_conditional[(0, 0)] == [1, 3, 8]
    assert model.pair_params_conditional[(0, 1)] == [2, 3, 4, 8]
    assert model.pair_params_conditional[(1, 1)] == [1, 2, 8]
    assert params.dataset_type == "vasp"

    train = parser.train
    test = parser.test
    assert train[0].name == "Train_dataset0/vasprun-*.xml.polymlp"
    assert train[0].include_force == True
    assert train[0].weight == pytest.approx(1.0)
    assert train[1].name == "Train_dataset1/vasprun-*.xml.polymlp"
    assert train[1].include_force == False
    assert train[1].weight == pytest.approx(0.1)
    assert train[2].name == "dataset2/vasprun-*.xml.polymlp"
    assert train[2].include_force == True
    assert train[2].weight == pytest.approx(1.0)
    assert train[3].name == "dataset3/vasprun-*.xml.polymlp"
    assert train[3].include_force == False
    assert train[3].weight == pytest.approx(0.1)
    assert train[4].name == "Train_dataset6/vasprun.xml"
    assert train[4].include_force == True
    assert train[4].weight == pytest.approx(1.0)
    assert train[5].name == "Train_dataset7/vasprun.xml"
    assert train[5].include_force == False
    assert train[5].weight == pytest.approx(0.1)

    assert test[0].name == "Test_dataset0/vasprun-*.xml.polymlp"
    assert test[0].include_force == True
    assert test[0].weight == pytest.approx(1.0)
    assert test[1].name == "Test_dataset1/vasprun-*.xml.polymlp"
    assert test[1].include_force == False
    assert test[1].weight == pytest.approx(0.1)
    assert test[2].name == "dataset4/vasprun-*.xml.polymlp"
    assert test[2].include_force == True
    assert test[2].weight == pytest.approx(1.0)
    assert test[3].name == "dataset5/vasprun-*.xml.polymlp"
    assert test[3].include_force == False
    assert test[3].weight == pytest.approx(0.1)
    assert test[4].name == "Test_dataset6/vasprun.xml"
    assert test[4].include_force == True
    assert test[4].weight == pytest.approx(1.0)
    assert test[5].name == "Test_dataset7/vasprun.xml"
    assert test[5].include_force == False
    assert test[5].weight == pytest.approx(0.1)
