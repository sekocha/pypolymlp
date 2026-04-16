"""Tests of API for thermodynamics calculations."""

import glob
import shutil
from pathlib import Path

import pytest

from pypolymlp.api.pypolymlp_thermodynamics import PypolymlpThermodynamics

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/others/thermodynamics/"

yamls_sscha = sorted(glob.glob(path_file + "sscha/*.yaml"))
yamls_el = glob.glob(path_file + "electrons/*.yaml")
yamls_el_ph = glob.glob(path_file + "electrons_sscha/*.yaml")


def test_PypolymlpThermodynamics():
    """Test PypolymlpThermodynamics."""
    api = PypolymlpThermodynamics(
        yamls_sscha=yamls_sscha,
        yamls_electron=yamls_el,
        yamls_electron_phonon=yamls_el_ph,
    )
    api.run()
    api.save(path="tmp")

    cp = api._thermo.sscha._eq_cp
    assert cp[5] == pytest.approx(24.94523630348574)

    cp = api._thermo.sscha_el._eq_cp
    assert cp[5] == pytest.approx(25.313507918844582)

    cp = api._thermo.sscha_el_ph._eq_cp
    assert cp[5] == pytest.approx(25.286377287737213)
    shutil.rmtree("tmp")


# TODO: Add test for PypolymlpTransion
