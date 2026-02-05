"""Tests of SSCHA parameter class."""

from pathlib import Path

import numpy as np

from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.interface_vasp import Poscar

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"

unitcell = Poscar(poscar).structure
size = (2, 2, 2)


def test_params():
    """Test SSCHAParams."""
    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temp_min=300,
        temp_max=500,
        temp_step=100,
    )
    np.testing.assert_equal(params.temperatures, [500, 400, 300])

    # temp: Optional[float] = None,
    # n_temp: Optional[int] = None,
    # ascending_temp: bool = False,
    # n_samples_init: Optional[int] = None,
    # n_samples_final: Optional[int] = None,
