"""Tests of EOS calculation using API."""

from pathlib import Path

from test_compute_eos import _assert_eos_MgO

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_eos_MgO():
    """Test EOS calculation."""
    poscar = path_file + "poscars/POSCAR.RS.idealMgO"
    pot = path_file + "mlps/polymlp.yaml.pair.MgO"

    polymlp = PypolymlpCalc(pot=pot, verbose=True)
    polymlp.load_poscars(poscar)
    polymlp.run_eos(
        eps_min=0.7,
        eps_max=2.0,
        eps_step=0.03,
        fine_grid=True,
        eos_fit=True,
    )
    e0, v0, b0 = polymlp.eos_fit_data
    _assert_eos_MgO(e0, v0, b0)
