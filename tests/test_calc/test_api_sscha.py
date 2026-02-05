"""Tests of SSCHA calculations using API."""

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
    sscha.run(temp=700, tol=0.003, mixing=0.5, path="tmp")
    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_sscha_Al_restart():
    """Test restart SSCHA calculations using API."""
    path_sscha = path_file + "others/sscha_restart/"
    yaml = path_sscha + "sscha_results.yaml"
    sscha = PypolymlpSSCHA(verbose=True)
    sscha.load_restart(yaml=yaml, parse_fc2=True, abspath=path_sscha)
    sscha.run(temp=700, tol=0.003, mixing=0.5, path="tmp")
    _assert_Al(sscha)
    shutil.rmtree("tmp")
