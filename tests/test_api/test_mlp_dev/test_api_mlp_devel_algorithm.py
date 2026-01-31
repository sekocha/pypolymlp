"""Tests of polynomial MLP development using API"""

from pathlib import Path

import pytest

from pypolymlp.core.interface_phono3py import Phono3pyYaml
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def _initialize(phono3py_mp_149):
    """Test sequential algorithm."""
    polymlp = Pypolymlp(verbose=True)
    polymlp.set_params(
        elements=["Si"],
        cutoff=6.0,
        model_type=3,
        max_p=2,
        gtinv_order=3,
        gtinv_maxl=[4, 4],
        gaussian_params2=[0.0, 4.0, 5],
        reg_alpha_params=(-1, 3, 5),
        atomic_energy=[0.0],
        include_stress=False,
    )

    ph3 = Phono3pyYaml(phono3py_mp_149)
    polymlp.set_datasets_structures_autodiv(
        structures=ph3.supercells,
        energies=ph3.energies,
        forces=ph3.forces,
    )
    return polymlp


def _assert(polymlp: Pypolymlp):
    """Assert errors."""
    error_train = polymlp.summary.error_train["data1"]
    error_test = polymlp.summary.error_test["data2"]
    print("Test")
    print(error_test)
    print("Train")
    print(error_train)

    assert error_test["energy"] == pytest.approx(2.7282304588694303e-06, rel=1e-3)
    assert error_test["force"] == pytest.approx(0.0011572393315208939, rel=1e-3)
    assert error_train["energy"] == pytest.approx(2.368768610001591e-06, rel=1e-3)
    assert error_train["force"] == pytest.approx(0.0011560576729359665, rel=1e-3)


def test_sequential(phono3py_mp_149):
    """Test sequential algorithm."""
    polymlp = _initialize(phono3py_mp_149)
    polymlp.fit()
    polymlp.estimate_error()
    _assert(polymlp)


def test_standard(phono3py_mp_149):
    """Test standard algorithm."""
    polymlp = _initialize(phono3py_mp_149)
    polymlp.fit_standard()
    polymlp.estimate_error()
    _assert(polymlp)


def test_cg(phono3py_mp_149):
    """Test CG algorithm."""
    polymlp = _initialize(phono3py_mp_149)
    polymlp.fit_cg(gtol=1e-10)
    polymlp.estimate_error()
    _assert(polymlp)


def test_run_sequential(phono3py_mp_149):
    """Test sequential algorithm using function run."""
    polymlp = _initialize(phono3py_mp_149)
    polymlp.run()
    _assert(polymlp)

    polymlp = _initialize(phono3py_mp_149)
    polymlp.run(batch_size=5)
    _assert(polymlp)


def test_run_cg(phono3py_mp_149):
    """Test CG algorithm using function run."""
    polymlp = _initialize(phono3py_mp_149)
    polymlp.run(use_cg=True, gtol=1e-10)
    _assert(polymlp)
