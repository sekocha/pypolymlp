"""Tests of SSCHA property calculation."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.properties import initialize_polymlp_calculator
from pypolymlp.calculator.sscha.api_properties import PropertiesSSCHA
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_sscha_properties_Al():
    """Test PropertiesSSCHA."""
    unitcell = Poscar(poscar).structure
    supercell_matrix = np.diag([2, 2, 2])
    prop = initialize_polymlp_calculator(pot=pot)
    sscha_params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=supercell_matrix,
        pot=prop.pot,
        temp=700,
        tol=0.01,
        mixing=0.5,
        use_mkl=False,
    )
    prop_sscha = PropertiesSSCHA(sscha_params, prop, verbose=True)
    free_energy, _, _ = prop_sscha.eval(unitcell)
    f_true = -14.346701887582924
    assert free_energy == pytest.approx(f_true, rel=1e-2)
    shutil.rmtree("sscha")

    assert tuple(prop_sscha.params.elements) == ("Al",)

    f_true_kj = -61.20517389569886
    assert prop_sscha.properties.free_energy == pytest.approx(f_true_kj, rel=1e-2)
    assert prop_sscha.logs[-1].free_energy == pytest.approx(f_true_kj, rel=1e-2)
    assert prop_sscha.force_constants.shape == (32, 32, 3, 3)
    assert prop_sscha.delta < 0.01
