"""Tests of polynomial MLP development using API"""

from pathlib import Path

import numpy as np
import pytest
from test_mlp_devel_phono3py import _check_errors_phono3py_yaml

from pypolymlp.core.interface_phono3py import Phono3pyYaml
from pypolymlp.core.utils import split_ids_train_test
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_api_phono3py():
    """Test API for MLP development using phono3py data."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Si"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 5.0, 6],
        atomic_energy=[0.0],
    )
    yaml = str(cwd / "data-phono3py-Si/phonopy_training_dataset.yaml.xz")
    polymlp.set_datasets_phono3py(yaml, train_ratio=0.9)
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    _check_errors_phono3py_yaml(error_train, error_test)


def test_mlp_devel_api_phono3py2(phono3py_mp_149):
    """Test API for MLP development using phono3py data."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Si"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 4.0, 5],
        atomic_energy=[0.0],
    )
    ph3 = Phono3pyYaml(phono3py_mp_149)
    polymlp.set_datasets_structures_autodiv(
        structures=ph3.supercells,
        energies=ph3.energies,
        forces=ph3.forces,
        train_ratio=0.9,
    )
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    assert error_test["energy"] == pytest.approx(2.3987675262637736e-06, abs=1e-8)
    assert error_test["force"] == pytest.approx(0.001078577920994468, rel=1e-3)
    assert error_train["energy"] == pytest.approx(2.229479437621172e-06, abs=1e-8)
    assert error_train["force"] == pytest.approx(0.0010762511671704963, rel=1e-3)


def test_mlp_devel_api_displacements(phono3py_mp_149):
    """Test API for MLP development using displacements."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Si"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 4.0, 5],
        atomic_energy=[0.0],
    )
    ph3 = Phono3pyYaml(phono3py_mp_149)
    energies = ph3.energies
    disps, forces = ph3.phonon_dataset

    train_ids, test_ids = split_ids_train_test(len(energies), train_ratio=0.9)
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
        ph3.supercell,
    )
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    assert error_test["energy"] == pytest.approx(2.3987675262637736e-06, abs=1e-8)
    assert error_test["force"] == pytest.approx(0.001078577920994468, rel=1e-3)
    assert error_train["energy"] == pytest.approx(2.229479437621172e-06, abs=1e-8)
    assert error_train["force"] == pytest.approx(0.0010762511671704963, rel=1e-3)


def test_mlp_devel_api_learning_curve(phono3py_mp_149):
    """Test API for learning curve."""
    polymlp = Pypolymlp(verbose=False)
    polymlp.set_params(
        elements=["Si"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 4.0, 5],
        atomic_energy=[0.0],
    )
    ph3 = Phono3pyYaml(phono3py_mp_149)
    n_max = 30
    polymlp.set_datasets_structures_autodiv(
        structures=ph3.supercells[:n_max],
        energies=ph3.energies[:n_max],
        forces=ph3.forces[:n_max],
        train_ratio=0.9,
    )
    polymlp.fit_learning_curve()
    error = polymlp.learning_curve

    assert error[0][1]["energy"] == pytest.approx(4.1273259261526104e-06, rel=1e-2)
    assert error[0][1]["force"] == pytest.approx(0.0013968999351407563, rel=1e-2)
    assert error[1][1]["energy"] == pytest.approx(2.8884836034348403e-06, rel=1e-2)
    assert error[1][1]["force"] == pytest.approx(0.0012583650861508371, rel=1e-2)
    assert error[2][1]["energy"] == pytest.approx(2.7201787938505056e-06, rel=1e-2)
    assert error[2][1]["force"] == pytest.approx(0.0012488734445901962, rel=1e-2)
    assert error[3][1]["energy"] == pytest.approx(3.4018165734520205e-06, rel=1e-2)
    assert error[3][1]["force"] == pytest.approx(0.0012411557673293271, rel=1e-2)
    assert error[4][1]["energy"] == pytest.approx(2.970098929885866e-06, rel=1e-2)
    assert error[4][1]["force"] == pytest.approx(0.0012207040687902018, rel=1e-2)
    assert error[5][1]["energy"] == pytest.approx(2.474675633675637e-06, rel=1e-2)
    assert error[5][1]["force"] == pytest.approx(0.001203376794506134, rel=1e-2)
    assert error[6][1]["energy"] == pytest.approx(2.2211487516731584e-06, rel=1e-2)
    assert error[6][1]["force"] == pytest.approx(0.0011839465316066475, rel=1e-2)
    assert error[7][1]["energy"] == pytest.approx(2.2940607346457032e-06, rel=1e-2)
    assert error[7][1]["force"] == pytest.approx(0.0011748749796290953, rel=1e-2)
    assert error[8][1]["energy"] == pytest.approx(2.4805390643169517e-06, rel=1e-2)
    assert error[8][1]["force"] == pytest.approx(0.0011733723065265286, rel=1e-2)
    assert error[9][1]["energy"] == pytest.approx(2.2567046723897206e-06, rel=1e-2)
    assert error[9][1]["force"] == pytest.approx(0.0011650521771290996, rel=1e-2)
    assert error[10][1]["energy"] == pytest.approx(2.2958264193043096e-06, rel=1e-2)
    assert error[10][1]["force"] == pytest.approx(0.0011650176624327495, rel=1e-2)
    assert error[11][1]["energy"] == pytest.approx(2.4497318148710586e-06, rel=1e-2)
    assert error[11][1]["force"] == pytest.approx(0.00116279394273854, rel=1e-2)
    assert error[12][1]["energy"] == pytest.approx(2.594962142892688e-06, rel=1e-2)
    assert error[12][1]["force"] == pytest.approx(0.0011614306531000504, rel=1e-2)


def _fit_assert_NaCl(polymlp):
    """Run fit in NaCl."""
    yamlfile = str(cwd / "data-phono3py-NaCl/phonopy_params_NaCl-rd.yaml.xz")
    train_ids = np.arange(8)
    test_ids = np.arange(8, 10)

    ph3 = Phono3pyYaml(yamlfile)
    disps, forces = ph3.phonon_dataset
    supercell = ph3.supercell
    energies = ph3.energies

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
        supercell,
    )
    polymlp.run()

    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    assert error_test["energy"] == pytest.approx(1.8219684368801306e-07, rel=1e-3)
    assert error_test["force"] == pytest.approx(0.0001582001857382102, rel=1e-3)
    assert error_train["energy"] == pytest.approx(3.1294935174049514e-07, rel=1e-3)
    assert error_train["force"] == pytest.approx(8.979273829689044e-05, rel=1e-3)


def test_mlp_devel_api_NaCl_structure1():
    """Test mlp development using phono3py.yaml and structure interface."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Cl", "Na"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 10],
        atomic_energy=[-0.245, -0.331],
    )
    _fit_assert_NaCl(polymlp)


def test_mlp_devel_api_NaCl_structure2():
    """Test mlp development using phono3py.yaml and structure interface."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Na", "Cl"],
        cutoff=8.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 7.0, 10],
        atomic_energy=[-0.331, -0.245],
    )
    _fit_assert_NaCl(polymlp)
