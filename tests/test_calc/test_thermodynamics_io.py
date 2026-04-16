"""Tests of thermodynamics_io."""

from pathlib import Path

from pypolymlp.calculator.thermodynamics.thermodynamics_io import (
    load_thermodynamics_yaml,
)

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"


def test_load_thermodynamics_yaml():
    """Test load_thermodynamics_yaml."""
    data = load_thermodynamics_yaml(path_file + "sscha.yaml")
    assert len(data.temperatures) == 16
    assert len(data.eq_volumes) == 16
    assert len(data.bm) == 16
    assert len(data.eq_helmholtz) == 16
    assert len(data.eq_entropy) == 16
    assert len(data.eq_cp) == 16
    assert len(data.eos_data) == 16
    assert len(data.eos_fit_data) == 16
    assert len(data.gibbs) == 16

    data1 = data.get_T_F()
    assert data1.shape == (16, 2)


# save_thermodynamics_yaml is tested in other functions.
