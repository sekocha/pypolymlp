"""Tests of SSCHA data class."""

import pytest

from pypolymlp.calculator.sscha.sscha_data import SSCHAData


def test_data():
    """Test SSCHAData."""
    data = SSCHAData(
        temperature=300,
        static_potential=0.1,
        harmonic_potential=0.2,
        harmonic_free_energy=0.3,
        average_potential=0.4,
        anharmonic_free_energy=0.5,
        entropy=0.6,
        harmonic_heat_capacity=0.7,
        static_forces=[0.1, 0.2],
        average_forces=[0.01, 0.02],
        delta=0.003,
        converge=True,
        imaginary=False,
    )
    assert data.free_energy == pytest.approx(0.8)
