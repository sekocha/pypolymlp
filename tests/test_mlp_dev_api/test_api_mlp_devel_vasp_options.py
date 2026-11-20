"""Tests of polynomial MLP development using API"""

import glob
from pathlib import Path

from test_mlp_devel_vasp_options import _check_errors_MgO, _check_errors_multi_MgO

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_single_MgO():
    """Test mlp development using API and single dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=2,
        max_p=2,
        feature_type="pair",
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=True,
        use_auto_weight=False,
        weight_force=0.8,
        weight_stress=1.0,
    )
    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    _check_errors_MgO(error_train, error_test)


def test_mlp_devel_api_multi_MgO():
    """Test mlp development using API and multiple datasets."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=2,
        max_p=2,
        feature_type="pair",
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=True,
        use_auto_weight=False,
        weight_force=0.8,
        weight_stress=1.0,
    )
    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    train_vaspruns2 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    test_vaspruns2 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test2/vasprun-*.xml.polymlp"
    )
    polymlp.set_multiple_datasets_vasp(
        [train_vaspruns1, train_vaspruns2], [test_vaspruns1, test_vaspruns2]
    )

    polymlp.run()
    error_train1 = polymlp.summary.error_train["data1"]
    error_train2 = polymlp.summary.error_train["data2"]
    error_test1 = polymlp.summary.error_test["data1"]
    error_test2 = polymlp.summary.error_test["data2"]

    _check_errors_multi_MgO(error_train1, error_train2, error_test1, error_test2)


def test_mlp_devel_api_single_cg_MgO():
    """Test mlp development using API and single dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=2,
        max_p=2,
        feature_type="pair",
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=True,
        solver_use_gradient=True,
        solver_gtol=1e-5,
        use_auto_weight=False,
        weight_force=0.8,
        weight_stress=1.0,
    )
    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    _check_errors_MgO(error_train, error_test)
