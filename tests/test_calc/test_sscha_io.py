"""Tests of IO class for SSCHA."""

import os

import numpy as np

from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.calculator.sscha.sscha_io import save_sscha_yaml
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams


def test_save_sscha_yaml(unitcell_mlp_Al):
    """Test save_sscha_yaml."""
    unitcell, pot = unitcell_mlp_Al
    size = (2, 2, 2)
    data = SSCHAData(
        temperature=300,
        static_potential=0.1,
        harmonic_potential=0.2,
        harmonic_free_energy=0.3,
        average_potential=0.4,
        anharmonic_free_energy=0.5,
        entropy=0.6,
        harmonic_heat_capacity=0.7,
        static_forces=np.array([[0.1, 0.2], [0.2, 0.3]]),
        average_forces=np.array([[0.01, 0.02], [0.2, 0.3]]),
        delta=0.003,
        converge=True,
        imaginary=False,
    )
    params = SSCHAParams(
        unitcell=unitcell,
        supercell_matrix=size,
        pot=pot,
        temp_min=300,
        temp_max=500,
        temp_step=100,
    )
    params.supercell = unitcell
    params.supercell.supercell_matrix = params.supercell_matrix
    save_sscha_yaml(params, [data], filename="tmp.yaml")
    os.remove("tmp.yaml")
