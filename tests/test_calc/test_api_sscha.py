"""Tests of SSCHA calculations using API."""

import os
import shutil
from pathlib import Path

from test_sscha_api_sscha import _assert_Al

from pypolymlp.api.pypolymlp_sscha import PypolymlpSSCHA

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_sscha_Al():
    """Test SSCHA calculations from polymlp using API."""
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(pot)
    sscha.run(temp=700, tol=0.003, mixing=0.5, path="tmp", use_mkl=False)
    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_sscha_Al_restart():
    """Test restart SSCHA calculations using API."""
    path_sscha = path_file + "others/sscha_restart/"
    yaml = path_sscha + "sscha_results.yaml"
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_restart(yaml=yaml, parse_fc2=True, pot=pot)
    sscha.run(temp=700, tol=0.003, mixing=0.5, path="tmp", use_mkl=False)
    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_sscha_geometry_opt():
    """Test SSCHA Geometry optimization."""
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(pot)

    sscha.run_geometry_optimization(
        temp=700,
        tol=0.02,
        mixing=0.5,
        use_mkl=False,
        gtol=1e-1,
        go_maxiter=2,
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        pressure=0.01,
    )
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")


def test_sscha_elastic():
    """Test SSCHA elastic constant calculation."""
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_poscar(poscar, (2, 2, 2))
    sscha.set_polymlp(pot)

    sscha.run_elastic(
        temp=300,
        tol=0.05,
        mixing=0.95,
        use_mkl=False,
        gtol=1e-1,
    )
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")
    os.remove("polymlp_elastic_sscha.yaml")
