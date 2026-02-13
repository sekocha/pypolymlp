"""Tests of polynomial MLP development using API"""

import glob
from pathlib import Path

import pytest
from test_mlp_devel_vasp import (
    _check_errors_md_Cu,
    _check_errors_multiple_datasets_MgO,
    _check_errors_single_dataset_MgO,
    _check_errors_single_dataset_MgO_auto,
)

from pypolymlp.core.interface_vasp import Vasprun, parse_structures_from_vaspruns
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_include_force_false():
    """Test mlp development using API."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_force=False,
        include_stress=False,
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)

    polymlp.run()
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(0.0010080932601856329, rel=1e-4)
    assert error_train1["energy"] == pytest.approx(4.205974853729061e-06, rel=1e-4)


def test_mlp_devel_api_single_dataset():
    """Test mlp development using API and single dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=False,
    )

    vaspruns1 = sorted(
        glob.glob(str(cwd) + "/data-vasp-MgO/vaspruns/*1/vasprun-*.xml.polymlp")
    )
    structures = parse_structures_from_vaspruns(vaspruns1)

    energies, forces = [], []
    for v in vaspruns1:
        e, f, s = Vasprun(v).properties
        energies.append(e)
        forces.append(f)

    polymlp.set_datasets_structures_autodiv(
        structures=structures, energies=energies, forces=forces, train_ratio=0.9
    )
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    _check_errors_single_dataset_MgO_auto(error_train, error_test)


def test_mlp_devel_api_single_dataset_element_swapped():
    """Test mlp development using API and single dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["O", "Mg"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-1.85321219, -0.00040000],
        include_stress=False,
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)

    polymlp.run()
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    _check_errors_single_dataset_MgO(error_train1, error_test1)


def test_mlp_devel_api_multidatasets():
    """Test mlp development using API and multiple datasets."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=False,
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

    _check_errors_multiple_datasets_MgO(
        error_train1, error_train2, error_test1, error_test2
    )


def test_mlp_devel_api_structure():
    """Test API for MLP development using structure dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=False,
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )

    train_structures = parse_structures_from_vaspruns(train_vaspruns1)
    test_structures = parse_structures_from_vaspruns(test_vaspruns1)

    train_energies, train_forces = [], []
    for v in train_vaspruns1:
        e, f, s = Vasprun(v).properties
        train_energies.append(e)
        train_forces.append(f)

    test_energies, test_forces = [], []
    for v in test_vaspruns1:
        e, f, s = Vasprun(v).properties
        test_energies.append(e)
        test_forces.append(f)

    polymlp.set_datasets_structures(
        train_structures=train_structures,
        test_structures=test_structures,
        train_energies=train_energies,
        test_energies=test_energies,
        train_forces=train_forces,
        test_forces=test_forces,
    )

    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]

    _check_errors_single_dataset_MgO(error_train, error_test, use_stress=False)

    polymlp.set_params(
        elements=["O", "Mg"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-1.85321219, -0.00040000],
    )
    polymlp.set_datasets_structures(
        train_structures=train_structures,
        test_structures=test_structures,
        train_energies=train_energies,
        test_energies=test_energies,
        train_forces=train_forces,
        test_forces=test_forces,
    )
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]

    _check_errors_single_dataset_MgO(error_train, error_test, use_stress=False)


def test_mlp_devel_api_structure_auto():
    """Test API for MLP development using structure dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
        include_stress=False,
    )

    vaspruns1 = sorted(
        glob.glob(str(cwd) + "/data-vasp-MgO/vaspruns/*1/vasprun-*.xml.polymlp")
    )
    structures = parse_structures_from_vaspruns(vaspruns1)

    energies, forces = [], []
    for v in vaspruns1:
        e, f, s = Vasprun(v).properties
        energies.append(e)
        forces.append(f)

    polymlp.set_datasets_structures_autodiv(
        structures=structures, energies=energies, forces=forces, train_ratio=0.9
    )
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    _check_errors_single_dataset_MgO_auto(error_train, error_test)


def test_mlp_devel_api_md():
    """Test API for MLP development using structure dataset."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Cu"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000],
        include_stress=False,
    )

    vaspruns = str(cwd) + "/data-vasp-md-Cu/vasprun.xml"
    polymlp.set_datasets_vasp(vaspruns=vaspruns)

    polymlp.run()
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    _check_errors_md_Cu(error_train1, error_test1)


def test_mlp_devel_api_distance():
    """Test MLP development using distance constraints."""
    polymlp = Pypolymlp(verbose=True)

    distance_dict = {
        ("Sr", "Sr"): [3.9, 5.5],
        ("Sr", "Ti"): [3.4, 6.5],
        ("Ti", "Ti"): [3.9, 5.5],
    }

    polymlp.set_params(
        elements=["Sr", "Ti", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.02805273, -2.44987314, -1.85321219],
        distance=distance_dict,
        include_stress=False,
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-SrTiO3/vaspruns/train1/vasprun.xml.*"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-vasp-SrTiO3/vaspruns/test1/vasprun.xml.*"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)

    polymlp.run()
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(0.0011914132092445697, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.02750490198874777, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(0.0015997025381622896, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01742941204519919, abs=1e-6)
