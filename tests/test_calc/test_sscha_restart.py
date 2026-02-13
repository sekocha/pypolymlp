"""Tests of restarting SSCHA calculations."""

from pathlib import Path

import pytest

from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.core.units import EVtoKJmol

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

pot = path_file + "mlps/polymlp.yaml.gtinv.Al"


def test_restart():
    """Test Restart class for SSCHA calculations."""
    path_sscha = path_file + "others/sscha_restart/"
    yaml = path_sscha + "sscha_results.yaml"
    fc2hdf5 = path_sscha + "fc2.hdf5"

    res = Restart(res_yaml=yaml, fc2hdf5=fc2hdf5, pot=pot, unit="kJ/mol")
    assert res.polymlp == pot
    assert res.temperature == 700

    free_energy = -61.134080255526314
    static_potential = -1322.92893961425
    entropy = 185.9304364307951
    harmonic_heat_capacity = 97.9516453918095
    anharmonic_free_energy = -2.1122313959897348

    assert res.free_energy == pytest.approx(free_energy)
    assert res.static_potential == pytest.approx(static_potential)
    assert res.entropy == pytest.approx(entropy)
    assert res.harmonic_heat_capacity == pytest.approx(harmonic_heat_capacity)
    assert res.anharmonic_free_energy == pytest.approx(anharmonic_free_energy)

    assert len(res.logs["free_energy"]) == 4
    assert len(res.logs["harmonic_potential"]) == 4
    assert len(res.logs["average_potential"]) == 4
    assert len(res.logs["anharmonic_free_energy"]) == 4

    assert res.delta_fc == pytest.approx(0.0038616104047306888)
    assert res.converge
    assert not res.imaginary

    assert res.force_constants.shape == (32, 32, 3, 3)
    assert res.unitcell.axis.shape == (3, 3)
    assert res.unitcell_phonopy is not None
    assert res.supercell_matrix.shape == (3, 3)
    assert res.n_unitcells == 8
    assert res.supercell.axis.shape == (3, 3)
    assert res.supercell_phonopy is not None
    assert res.volume == pytest.approx(65.77091478008525)

    n_atom = len(res.unitcell.elements)
    res.unit = "eV/atom"
    converter = 1.0 / (EVtoKJmol * n_atom)
    converter_J = 1.0 / (EVtoKJmol * n_atom * 1000)
    assert res.free_energy == pytest.approx(free_energy * converter)
    assert res.static_potential == pytest.approx(static_potential * converter)
    assert res.entropy == pytest.approx(entropy * converter_J)
    assert res.harmonic_heat_capacity == pytest.approx(
        harmonic_heat_capacity * converter_J
    )
    assert res.anharmonic_free_energy == pytest.approx(
        anharmonic_free_energy * converter
    )

    res.unit = "eV/cell"
    converter = 1.0 / EVtoKJmol
    converter_J = 1.0 / (EVtoKJmol * 1000)
    assert res.free_energy == pytest.approx(free_energy * converter)
    assert res.static_potential == pytest.approx(static_potential * converter)
    assert res.entropy == pytest.approx(entropy * converter_J)
    assert res.harmonic_heat_capacity == pytest.approx(
        harmonic_heat_capacity * converter_J
    )
    assert res.anharmonic_free_energy == pytest.approx(
        anharmonic_free_energy * converter
    )
