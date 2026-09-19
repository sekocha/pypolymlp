"""Tests of polynomial MLP development using online algorithm"""

import glob
import os
from pathlib import Path

from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

cwd = Path(__file__).parent


def test_mlp_devel_online():
    """Run fitting using online algorithm."""
    pypolymlp = Pypolymlp()
    file = str(cwd) + "/mlps/polymlp.yaml.gtinv"
    pypolymlp.load_mlp(file, require_atomic_energy=True)

    test = str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    vaspruns = glob.glob(test)
    pypolymlp.set_datasets_vasp_online(vaspruns=vaspruns)

    pypolymlp.fit_online(max_learning_rate=1e-5, n_epochs=3)
    assert len(pypolymlp.coeffs) == 1899
    assert len(pypolymlp.summary.scaled_coeffs) == 1899
    pypolymlp.save_mlp("tmp.yaml")
    os.remove("tmp.yaml")


def test_mlp_devel_hybrid_online():
    """Run fitting hybrid MLP using online algorithm."""
    pypolymlp = Pypolymlp()
    file = str(cwd) + "/mlps/polymlp.yaml.gtinv"
    pypolymlp.load_mlp([file, file], require_atomic_energy=True)

    test = str(cwd) + "/data-vasp-MgO/vaspruns/test1/vasprun-*.xml.polymlp"
    vaspruns = glob.glob(test)
    pypolymlp.set_datasets_vasp_online(vaspruns=vaspruns)

    pypolymlp.fit_online(
        max_learning_rate=1e-5,
        alpha=0.5,
        beta=0.92,
        gtol=1e-3,
        n_epochs=3,
    )
    assert len(pypolymlp.coeffs) == 2
    assert len(pypolymlp.coeffs[0]) == 1899
    assert len(pypolymlp.coeffs[1]) == 1899

    pypolymlp.save_mlp("tmp.yaml")
    os.remove("tmp.yaml.1")
    os.remove("tmp.yaml.2")
