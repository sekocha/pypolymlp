"""Tests of vasp utility functions."""

from pathlib import Path

import numpy as np

from pypolymlp.utils.vasp_utils import (
    load_electronic_properties_from_vasprun,
    print_poscar,
)

cwd = Path(__file__).parent


def test_electronic_properties():
    """Test for load_electronic_properties_from_vasprun."""
    filename = cwd / "./../files/vasprun-00001-Ti-full.xml"
    st, el = load_electronic_properties_from_vasprun(
        input_filename=filename, temp_max=30, temp_step=10
    )
    np.testing.assert_allclose(
        el.energy,
        [0.0, 0.00012229, 0.00051926, 0.00125968],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        el.entropy,
        [0, 2.92224116e-05, 5.55669080e-05, 8.50459424e-05],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        el.free_energy,
        [0.0, -0.00016993, -0.00059208, -0.00129169],
        atol=1e-6,
    )
    np.testing.assert_allclose(el.temperatures, [0.0, 10.0, 20.0, 30.0], atol=1e-6)


def test_print_poscar(structure_rocksalt):
    """Test print_poscar."""
    print_poscar(structure_rocksalt)
