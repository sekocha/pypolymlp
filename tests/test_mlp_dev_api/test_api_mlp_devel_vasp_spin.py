"""Tests of polynomial MLP development using API and spin data."""

import glob
from pathlib import Path

from test_mlp_devel_vasp_spin import _check_errors_hybrid_spin_Fe, _check_errors_spin_Fe

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent

train_vaspruns = glob.glob(str(cwd) + "/data-vasp-spin-Fe/vaspruns/train/*.xml")
test_vaspruns = glob.glob(str(cwd) + "/data-vasp-spin-Fe/vaspruns/test/*.xml")


def test_mlp_dev_spin_Fe():
    """Test MLP development using spin-polarized data."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Fe"],
        cutoff=6.0,
        model_type=2,
        max_p=3,
        feature_type="pair",
        reg_alpha_params=(-4, 1, 6),
        n_gaussians=9,
        atomic_energy=(-3.37684106,),
        enable_spins=(True,),
    )
    polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
    polymlp.run()

    assert polymlp.n_features == 2439
    tag_train = "data1"
    tag_test = "data2"
    error_train = polymlp.summary.error_train[tag_train]
    error_test = polymlp.summary.error_test[tag_test]
    _check_errors_spin_Fe(error_train, error_test)


def test_mlp_dev_hybrid_spin_Fe():
    """Test hybrid MLP development using spin-polarized data."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Fe"],
        cutoff=6.0,
        model_type=2,
        max_p=3,
        feature_type="pair",
        reg_alpha_params=(-4, 1, 6),
        n_gaussians=9,
        atomic_energy=(-3.37684106,),
        enable_spins=(True,),
    )
    polymlp.append_hybrid_params(
        elements=["Fe"],
        cutoff=3.0,
        model_type=2,
        max_p=2,
        feature_type="pair",
        n_gaussians=5,
        enable_spins=(True,),
    )
    polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
    polymlp.run()

    assert polymlp.n_features == 2549
    tag_train = "data1"
    tag_test = "data2"
    error_train = polymlp.summary.error_train[tag_train]
    error_test = polymlp.summary.error_test[tag_test]
    _check_errors_hybrid_spin_Fe(error_train, error_test)
