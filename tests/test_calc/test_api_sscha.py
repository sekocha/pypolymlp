"""Tests of SSCHA calculations using API."""

import os
import shutil
from pathlib import Path

from test_sscha_api_sscha import _assert_Al

from pypolymlp.api.api_calculator import PypolymlpCalcProperties
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_sscha_Al():
    """Test SSCHA calculations from polymlp using API."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_polymlp(pot=pot)
    unitcell = polymlp.load_poscar(poscar)
    prop = polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=(2, 2, 2),
        temp=700,
        tol=0.003,
        mixing=0.5,
        path="tmp",
        use_mkl=False,
    )
    calc = PypolymlpCalc(properties=prop, verbose=True)
    calc.eval(unitcell)
    sscha = calc._prop._sscha

    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_sscha_Al_grad():
    """Test SSCHA calculations from polymlp using API."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_polymlp(pot=pot)
    unitcell = polymlp.load_poscar(poscar)
    prop = polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=(2, 2, 2),
        temp=700,
        tol=0.003,
        mixing=0.5,
        path="tmp",
        precondition=False,
        use_mkl=False,
        symfc_batch_size=1000,
        symfc_use_gradient_solver=True,
    )
    calc = PypolymlpCalc(properties=prop, verbose=True)
    calc.eval(unitcell)
    sscha = calc._prop._sscha

    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_sscha_Al_restart():
    """Test restart SSCHA calculations using API."""
    yaml = path_file + "others/sscha_restart/sscha_results.yaml"
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.load_sscha_restart(yaml, pot=pot, parse_fc2=True)

    polymlp.set_sscha_calculator(
        temp=700,
        tol=0.003,
        mixing=0.5,
        path="tmp",
        use_mkl=False,
    )
    polymlp.eval(polymlp.sscha_unitcell)
    sscha = polymlp.calculator._sscha

    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_sscha_geometry_opt():
    """Test SSCHA Geometry optimization."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_polymlp(pot=pot)
    unitcell = polymlp.load_poscar(poscar)
    prop = polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=(2, 2, 2),
        temp=700,
        tol=0.02,
        mixing=0.5,
        path="tmp",
        use_mkl=False,
    )

    calc = PypolymlpCalc(properties=prop, verbose=True)
    calc.structures = unitcell
    calc.init_geometry_optimization(
        relax_cell=True,
        relax_volume=True,
        relax_positions=True,
        pressure=0.01,
    )
    calc.run_geometry_optimization(gtol=1e-1, maxiter=2)
    calc.save_poscars(filename="tmp_POSCAR")
    shutil.rmtree("tmp")
    os.remove("tmp_POSCAR")


def test_sscha_elastic():
    """Test SSCHA elastic constant calculation."""
    polymlp = PypolymlpCalcProperties(verbose=True)
    polymlp.set_polymlp(pot=pot)
    unitcell = polymlp.load_poscar(poscar)
    prop = polymlp.set_sscha_calculator(
        unitcell=unitcell,
        supercell_matrix=(2, 2, 2),
        temp=300,
        tol=0.05,
        mixing=0.95,
        path="tmp",
        use_mkl=False,
    )

    calc = PypolymlpCalc(properties=prop, verbose=True)
    calc.structures = unitcell
    calc.run_elastic_constants_temperature(gtol=0.1)
    calc.save_poscars(filename="tmp_POSCAR")
    calc.write_elastic_constants(filename="tmp.yaml")

    shutil.rmtree("tmp")
    os.remove("POSCAR_eqm")
    os.remove("tmp_POSCAR")
    os.remove("tmp.yaml")
