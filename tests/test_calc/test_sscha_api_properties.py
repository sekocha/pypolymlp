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
    assert free_energy == pytest.approx(-14.346701887582924, rel=1e-3)
    shutil.rmtree("sscha")

    assert tuple(prop_sscha.params.elements) == ("Al",)
