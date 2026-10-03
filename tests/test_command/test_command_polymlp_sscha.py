"""Test func_sscha for APIs."""

import os
import shutil
from pathlib import Path

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.func_sscha import run_main_sscha
from pypolymlp.api.run_polymlp_sscha import _parse_args_pypolymlp_sscha

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"

polymlp = PypolymlpCalcProperties(verbose=True)
polymlp.set_polymlp(pot=pot)


def test_run_functions1():
    """Test func_sscha."""
    args = _parse_args_pypolymlp_sscha([])
    args.poscar = poscar
    args.temp = 300
    args.tol = 0.07
    run_main_sscha(args, polymlp)
    shutil.rmtree("sscha")


def test_run_functions_go():
    """Test func_sscha."""
    args = _parse_args_pypolymlp_sscha(["--geometry_optimization"])
    args.poscar = poscar
    args.temp = 300
    args.gtol = 0.1
    args.tol = 0.07
    run_main_sscha(args, polymlp)
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")


def test_run_functions_eos():
    """Test func_sscha."""
    args = _parse_args_pypolymlp_sscha(["--eos"])
    args.poscar = poscar
    args.temp = 300
    args.tol = 0.07
    run_main_sscha(args, polymlp)
    shutil.rmtree("sscha")
    os.remove("polymlp_eos.yaml")


def test_run_functions_elastic():
    """Test func_sscha."""
    args = _parse_args_pypolymlp_sscha(["--elastic"])
    args.poscar = poscar
    args.temp = 300
    args.gtol = 0.1
    args.tol = 0.07
    run_main_sscha(args, polymlp)
    shutil.rmtree("sscha")
    os.remove("POSCAR_eqm")
    os.remove("polymlp_elastic_sscha.yaml")


# def test_run_functions_gsfe():
#     """Test func_sscha."""
#     args = _parse_args_pypolymlp_sscha(["--gsfe"])
#     args.poscar = poscar
#     args.temp = 300
#     args.gtol = 0.1
#     args.tol = 0.07
#     args.disp1 = (1, 0, -1)
#     args.disp2 = (1, -2, 1)
#     args.slip = (1, 1, 1)
#     args.n_layer = 1
#     args.n_points = 1
#     try:
#         run_main_sscha(args, polymlp)
#     except RuntimeError:
#         pass
#
#     shutil.rmtree("sscha")
#     shutil.rmtree("poscars")
#     os.remove("gsfe.dat")
