"""Test func_calc for APIs."""

import glob
import os
import shutil
from pathlib import Path

import pytest

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.developer.run_lammps_calc import _parse_args_lammps_calc
from pypolymlp.api.func_calc import run_calculations
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"

pytest.importorskip("lammps")
api = PypolymlpCalcProperties()
prop = api.set_lammps(elements=("Ag",), pot=pot)

calc = PypolymlpCalc(properties=prop, verbose=True)
calc.load_poscars(poscar)


def test_run_properties():
    args = _parse_args_lammps_calc(["--properties", "--pot", pot, "--elements", "Ag"])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc, calc_features=False)
    for f in glob.glob("polymlp_*"):
        os.remove(f)


def test_geometry_optimizations():
    args = _parse_args_lammps_calc(
        ["--geometry_optimization", "--pot", pot, "--elements", "Ag"]
    )
    args.poscar = args.poscars = poscar
    run_calculations(args, calc, calc_features=False)
    os.remove("POSCAR_eqm")


def test_run_eos():
    args = _parse_args_lammps_calc(["--eos", "--pot", pot, "--elements", "Ag"])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc, calc_features=False)
    os.remove("polymlp_eos.yaml")


def test_run_elastic():
    args = _parse_args_lammps_calc(["--elastic", "--pot", pot, "--elements", "Ag"])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc, calc_features=False)
    os.remove("polymlp_elastic.yaml")


def test_run_phonon():
    args = _parse_args_lammps_calc(["--phonon", "--pot", pot, "--elements", "Ag"])
    args.poscar = args.poscars = poscar
    args.mesh = (2, 2, 2)
    run_calculations(args, calc, calc_features=False)
    os.remove("polymlp_phonon.yaml")
    shutil.rmtree("polymlp_phonon_qha")
    for f in glob.glob("phonon*"):
        os.remove(f)


def test_gsfe():
    args = _parse_args_lammps_calc(["--gsfe", "--pot", pot, "--elements", "Ag"])
    args.poscar = args.poscars = poscar
    args.disp1 = (1, 0, 0)
    args.disp2 = (0, 1, 0)
    args.slip = (0, 0, 1)
    run_calculations(args, calc, calc_features=False)
    os.remove("gsfe.dat")
    shutil.rmtree("poscars")


def test_run_force_constants():
    args = _parse_args_lammps_calc(
        ["--force_constants", "--pot", pot, "--elements", "Ag"]
    )
    args.poscar = args.poscars = poscar
    run_calculations(args, calc, calc_features=False)
    os.remove("fc2.hdf5")
    os.remove("fc3.hdf5")
