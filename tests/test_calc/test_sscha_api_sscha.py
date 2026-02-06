"""Tests of API for SSCHA functions."""

import shutil
from pathlib import Path

import pytest

from pypolymlp.calculator.sscha.api_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"

unitcell = Poscar(poscar).structure
size = (2, 2, 2)


def _assert_Al(sscha):
    """Assert results for Al."""
    _ = sscha.sscha_params
    props = sscha.properties
    _ = sscha.logs
    fc2 = sscha.force_constants
    assert fc2.shape == (32, 32, 3, 3)

    assert props.static_potential == pytest.approx(-1322.92893961425, rel=1e-8)
    assert props.harmonic_potential == pytest.approx(34.50552229486996, rel=1e-2)
    assert props.harmonic_free_energy == pytest.approx(-59.06006839696035, rel=1e-2)
    assert props.average_potential == pytest.approx(32.438167553114006, rel=1e-2)
    assert props.anharmonic_free_energy == pytest.approx(-2.0673547417559606, rel=1e-1)
    assert props.free_energy == pytest.approx(-61.12742313871631, rel=1e-2)
    assert props.entropy == pytest.approx(185.98197787014593, rel=1e-2)
    assert props.harmonic_heat_capacity == pytest.approx(97.95462787508784, rel=1e-2)
    assert props.delta < 0.003
    assert props.converge
    assert not props.imaginary


def test_run_sscha():
    """Test run_sscha."""
    sscha_params = SSCHAParams(unitcell, size, pot=pot, temp=700, tol=0.003)
    sscha = run_sscha(sscha_params, pot=pot, path="tmp")
    _assert_Al(sscha)
    shutil.rmtree("tmp")


def test_run_sscha2():
    """Test run_sscha."""
    sscha_params = SSCHAParams(unitcell, size, pot=pot, temp=700, tol=0.003)
    sscha = run_sscha(
        sscha_params,
        pot=pot,
        use_temporal_cutoff=True,
        path="tmp",
        write_pdos=True,
    )
    _assert_Al(sscha)
    shutil.rmtree("tmp")
