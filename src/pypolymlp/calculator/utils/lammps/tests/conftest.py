"""Pytest conftest.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps

cwd = Path(__file__).parent


@pytest.fixture(scope="session")
def property_mlp_Ti_Al():
    """Return property lammps instance for polymlp in Ti-Al."""
    path = str(cwd) + "/files/Ti-Al/"
    pot = path + "/polymlp.lammps"
    elements = ["Ti", "Al"]
    prop = PropertiesLammps(
        elements=elements,
        pot=pot,
        style="polymlp",
        style_command="pair_style",
        coeff_command="pair_coeff",
        log=False,
        verbose=False,
    )
    return prop
