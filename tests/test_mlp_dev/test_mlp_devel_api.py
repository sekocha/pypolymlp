"""Tests of polynomial MLP development using API"""

import glob
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.core.interface_phono3py import Phono3pyYaml
from pypolymlp.core.interface_vasp import Vasprun, parse_structures_from_vaspruns
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_include_force_false():
    polymlp = Pypolymlp()
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
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)

    polymlp.run(verbose=True)
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(0.0010080932601856329, rel=1e-4)
    assert error_train1["energy"] == pytest.approx(4.205974853729061e-06, rel=1e-4)


def test_mlp_devel_api_single_dataset():
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)

    polymlp.run(verbose=True)
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(5.7907010720826916e-05, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.0048161516715470466, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.015092685843715342, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(3.149610950737406e-05, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.003828784221661806, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.015299231964527185, abs=1e-5)


def test_mlp_devel_api_multidatasets():
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    train_vaspruns2 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/train2/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    )
    test_vaspruns2 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/test2/vasprun-*.xml.polymlp"
    )
    polymlp.set_multiple_datasets_vasp(
        [train_vaspruns1, train_vaspruns2], [test_vaspruns1, test_vaspruns2]
    )

    polymlp.run(verbose=True)
    error_train1 = polymlp.summary.error_train["dataset1"]
    error_train2 = polymlp.summary.error_train["dataset2"]
    error_test1 = polymlp.summary.error_test["dataset1"]
    error_test2 = polymlp.summary.error_test["dataset2"]

    assert error_test1["energy"] == pytest.approx(2.2913988829580454e-4, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.01565802222609203, abs=1e-6)
    assert error_test1["stress"] == pytest.approx(0.01109818, abs=1e-5)

    assert error_test2["energy"] == pytest.approx(6.022423646649027e-3, abs=1e-8)
    assert error_test2["force"] == pytest.approx(0.40361027823620954, abs=1e-6)
    assert error_test2["stress"] == pytest.approx(0.03756416, abs=1e-5)

    assert error_train1["energy"] == pytest.approx(2.7039999690544686e-4, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.015516294909701862, abs=1e-6)
    assert error_train1["stress"] == pytest.approx(0.01024817, abs=1e-5)

    assert error_train2["energy"] == pytest.approx(6.587197553779358e-3, abs=1e-8)
    assert error_train2["force"] == pytest.approx(0.3157543533276883, abs=1e-6)
    assert error_train2["stress"] == pytest.approx(0.03763428, abs=1e-5)


def test_mlp_devel_api_structure():
    """Test API for MLP development using structure dataset."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Mg", "O"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 8],
        atomic_energy=[-0.00040000, -1.85321219],
    )

    train_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/train1/vasprun-*.xml.polymlp"
    )
    test_vaspruns1 = glob.glob(
        str(cwd) + "/data-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
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

    polymlp.run(verbose=True)

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]

    assert error_test["energy"] == pytest.approx(5.953749613283076e-5, abs=1e-8)
    assert error_test["force"] == pytest.approx(0.0048235814573149415, abs=1e-6)
    assert error_train["energy"] == pytest.approx(3.1366509877054183e-05, abs=1e-8)
    assert error_train["force"] == pytest.approx(0.003831185313049865, abs=1e-6)

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
    polymlp.run(verbose=True)

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]

    assert error_test["energy"] == pytest.approx(5.953749613283076e-5, abs=1e-8)
    assert error_test["force"] == pytest.approx(0.0048235814573149415, abs=1e-6)
    assert error_train["energy"] == pytest.approx(3.1366509877054183e-05, abs=1e-8)
    assert error_train["force"] == pytest.approx(0.003831185313049865, abs=1e-6)


def test_mlp_devel_api_phono3py():
    """Test API for MLP development using phono3py format."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Ag", "I"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 10],
        atomic_energy=[-0.19820116, -0.21203241],
    )
    yaml = cwd / "data-AgI/phono3py_params_wurtzite_AgI.yaml.xz"
    energy_dat = cwd / "data-AgI/energies_ltc_wurtzite_AgI_fc3-forces.dat"

    polymlp.set_datasets_phono3py(yaml, energy_dat=energy_dat)
    polymlp.run(verbose=True)

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]

    assert error_test["energy"] == pytest.approx(7.111637291840033e-07, rel=1e-4)
    assert error_test["force"] == pytest.approx(0.00015272719503135675, rel=1e-3)
    assert error_train["energy"] == pytest.approx(6.913212186834068e-07, rel=1e-4)
    assert error_train["force"] == pytest.approx(0.00015231860444491906, rel=1e-3)


def test_mlp_devel_api_displacements():
    """Test API for MLP development using displacements."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Ag", "I"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 10],
        atomic_energy=[-0.19820116, -0.21203241],
    )
    yamlfile = cwd / "data-AgI/phono3py_params_wurtzite_AgI.yaml.xz"
    energy_dat = cwd / "data-AgI/energies_ltc_wurtzite_AgI_fc3-forces.dat"
    train_ids = np.arange(5)
    test_ids = np.arange(95, 100)

    energies = np.loadtxt(energy_dat)[1:, 1]

    ph3 = Phono3pyYaml(yamlfile)
    disps, forces = ph3.phonon_dataset
    st_dict, _ = ph3.structure_dataset

    train_disps = disps[train_ids]
    train_forces = forces[train_ids]
    train_energies = energies[train_ids]
    test_disps = disps[test_ids]
    test_forces = forces[test_ids]
    test_energies = energies[test_ids]

    polymlp.set_datasets_displacements(
        train_disps,
        train_forces,
        train_energies,
        test_disps,
        test_forces,
        test_energies,
        st_dict,
    )
    polymlp.run(verbose=True)

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]

    assert error_test["energy"] == pytest.approx(4.3874684010604975e-07, abs=1e-8)
    assert error_test["force"] == pytest.approx(0.0002726224834356184, rel=1e-3)
    assert error_train["energy"] == pytest.approx(5.340620968588559e-07, abs=1e-8)
    assert error_train["force"] == pytest.approx(0.0001243130621149701, rel=1e-3)


def test_mlp_devel_api_learning_curve():
    """Test API for learning curve."""
    polymlp = Pypolymlp()
    polymlp.set_params(
        elements=["Ag", "I"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 10],
        atomic_energy=[-0.19820116, -0.21203241],
    )
    yamlfile = cwd / "data-AgI/phono3py_params_wurtzite_AgI.yaml.xz"
    energy_dat = cwd / "data-AgI/energies_ltc_wurtzite_AgI_fc3-forces.dat"
    train_ids = np.arange(30)
    test_ids = np.arange(95, 100)

    energies = np.loadtxt(energy_dat)[1:, 1]

    ph3 = Phono3pyYaml(yamlfile)
    disps, forces = ph3.phonon_dataset
    st_dict, _ = ph3.structure_dataset

    train_disps = disps[train_ids]
    train_forces = forces[train_ids]
    train_energies = energies[train_ids]
    test_disps = disps[test_ids]
    test_forces = forces[test_ids]
    test_energies = energies[test_ids]

    polymlp.set_datasets_displacements(
        train_disps,
        train_forces,
        train_energies,
        test_disps,
        test_forces,
        test_energies,
        st_dict,
    )
    polymlp.fit_learning_curve(verbose=True)
    error = polymlp.learning_curve.error
    assert error[0][1]["energy"] == pytest.approx(1.4029702831658057e-06, rel=1e-4)
    assert error[0][1]["force"] == pytest.approx(0.0003922588310867841, rel=1e-4)
    assert error[1][1]["energy"] == pytest.approx(4.314997208182684e-07, rel=1e-4)
    assert error[1][1]["force"] == pytest.approx(0.0002515380459997143, rel=1e-4)
    assert error[2][1]["energy"] == pytest.approx(4.543747990609749e-07, rel=1e-4)
    assert error[2][1]["force"] == pytest.approx(0.00023013652358275415, rel=1e-4)
    assert error[3][1]["energy"] == pytest.approx(3.33215986567445e-07, rel=1e-4)
    assert error[3][1]["force"] == pytest.approx(0.00020448702676183396, rel=1e-4)
    assert error[4][1]["energy"] == pytest.approx(3.6145457382140355e-07, rel=1e-4)
    assert error[4][1]["force"] == pytest.approx(0.0001917046038271312, rel=1e-4)
    assert error[5][1]["energy"] == pytest.approx(2.2370505273003928e-07, rel=1e-4)
    assert error[5][1]["force"] == pytest.approx(0.00018210848429612217, rel=1e-4)
    assert error[6][1]["energy"] == pytest.approx(3.042197443410954e-07, rel=1e-4)
    assert error[6][1]["force"] == pytest.approx(0.00017835747959455314, rel=1e-4)
    assert error[7][1]["energy"] == pytest.approx(3.5647510435178803e-07, rel=1e-4)
    assert error[7][1]["force"] == pytest.approx(0.0001753667685823238, rel=1e-4)
    assert error[8][1]["energy"] == pytest.approx(3.3211905496495353e-07, rel=1e-4)
    assert error[8][1]["force"] == pytest.approx(0.00017153251027763767, rel=1e-4)
    assert error[9][1]["energy"] == pytest.approx(3.386705988797691e-07, rel=1e-4)
    assert error[9][1]["force"] == pytest.approx(0.0001690526296815278, rel=1e-4)


def test_mlp_devel_api_distance():

    polymlp = Pypolymlp()

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
    )

    train_vaspruns1 = glob.glob(str(cwd) + "/data-SrTiO3/vaspruns/train1/vasprun.xml.*")
    test_vaspruns1 = glob.glob(str(cwd) + "/data-SrTiO3/vaspruns/test1/vasprun.xml.*")
    polymlp.set_datasets_vasp(train_vaspruns1, test_vaspruns1)

    polymlp.run(verbose=True)
    error_train1 = polymlp.summary.error_train["data1"]
    error_test1 = polymlp.summary.error_test["data2"]

    assert error_test1["energy"] == pytest.approx(0.0011914132092445697, abs=1e-8)
    assert error_test1["force"] == pytest.approx(0.02750490198874777, abs=1e-6)
    assert error_train1["energy"] == pytest.approx(0.0015997025381622896, abs=1e-8)
    assert error_train1["force"] == pytest.approx(0.01742941204519919, abs=1e-6)
