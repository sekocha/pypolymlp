"""Test func_calc for APIs."""

import glob
import os
import shutil
from pathlib import Path

from pypolymlp.api.func_calc import (
    run_calculations,
    run_elastic,
    run_eos,
    run_geometry_optimization,
    run_gsfe,
    run_phonon,
)
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.api.run_polymlp_calc import _parse_args_pypolymlp_calc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Al"
pot = path_file + "polymlp.yaml.gtinv.Al"


calc = PypolymlpCalc(pot=pot, verbose=True)
calc.load_poscars(poscar)
unitcell = calc.first_structure


def test_run_functions():
    """Test func_calc."""
    args = _parse_args_pypolymlp_calc([])
    args.poscar = args.poscars = poscar

    run_geometry_optimization(args, calc)
    run_geometry_optimization(args, calc, structure=unitcell)

    run_elastic(args, calc)
    run_elastic(args, calc, structure=unitcell)

    run_eos(args, calc)
    run_eos(args, calc, structure=unitcell)

    run_phonon(args, calc)

    args.disp1 = (1, 0, 0)
    args.disp2 = (0, 1, 0)
    args.slip = (0, 0, 1)
    run_gsfe(args, calc)

    os.remove("POSCAR_eqm")
    os.remove("fc2.hdf5")
    os.remove("gsfe.dat")
    os.remove("polymlp_elastic.yaml")
    os.remove("polymlp_eos.yaml")
    os.remove("polymlp_phonon.yaml")
    shutil.rmtree("polymlp_phonon_qha")
    shutil.rmtree("poscars")
    for f in glob.glob("phonon*"):
        os.remove(f)


def test_run_properties():
    args = _parse_args_pypolymlp_calc(["--properties", "--pot", pot])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc)
    for f in glob.glob("polymlp_*"):
        os.remove(f)


def test_run_features():
    """Test func_calc."""
    args = _parse_args_pypolymlp_calc(["--features", "--pot", pot])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc)
    os.remove("features.npy")


def test_run_force_constants():
    args = _parse_args_pypolymlp_calc(["--force_constants", "--pot", pot])
    args.poscar = args.poscars = poscar
    run_calculations(args, calc)
    os.remove("fc2.hdf5")
    os.remove("fc3.hdf5")


def test_gsfe():
    args = _parse_args_pypolymlp_calc(["--gsfe", "--pot", pot])
    args.poscar = args.poscars = poscar
    args.disp1 = (1, 0, 0)
    args.disp2 = (0, 1, 0)
    args.slip = (0, 0, 1)
    run_calculations(args, calc)
    os.remove("gsfe.dat")
    shutil.rmtree("poscars")
