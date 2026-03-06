"""Tests of functions to draw figures for properties."""

import shutil
from pathlib import Path

import numpy as np

from pypolymlp.calculator.auto.autocalc_distribution import AutoCalcDistribution

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_AutoCalcDistribution1():
    """Test AutoCalcDistribution for elemental system."""
    system = "Ag"
    pot_id = "polymlp-00001"
    vaspruns = [
        path_file + "others/vasprun-00001-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
    ]
    icsd_ids = ["104296", "105489", "105489"]

    api = AutoCalcDistribution(
        pot=path_file + "mlps/polymlp.lammps.pair.Ag",
        path_output="tmp",
    )
    api.compare_with_dft(vaspruns=vaspruns, icsd_ids=icsd_ids)
    api.plot_comparison_with_dft(system, pot_id)

    api.calc_energy_distribution(vaspruns, vaspruns)
    api.plot_energy_distribution(system, pot_id)
    shutil.rmtree("tmp")


def test_AutoCalcDistribution2():
    """Test AutoCalcDistribution for binary alloy system."""

    system = "Ti-Al"
    pot_id = "polymlp-00001"
    vaspruns = [
        path_file + "others/vasprun-00001-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
        path_file + "others/vasprun-end-Ti.xml",
        path_file + "others/vasprun-end-Al.xml",
    ]
    icsd_ids = ["99787-01", "99787-10", "99787-10", "652876", "52914"]

    api = AutoCalcDistribution(
        pot=path_file + "mlps/polymlp.lammps.gtinv.Ti-Al",
        path_output="tmp",
    )
    api.compare_with_dft(vaspruns=vaspruns, icsd_ids=icsd_ids)
    api.plot_comparison_with_dft(system, pot_id)

    api.calc_formation_energies(
        vaspruns=vaspruns,
        icsd_ids=icsd_ids,
        geometry_optimization=False,
    )
    api.plot_binary_formation_energies(system, pot_id)

    api.calc_energy_distribution(vaspruns, vaspruns)
    api.plot_energy_distribution(system, pot_id)
    shutil.rmtree("tmp")


def test_EnergyData():
    """Test EnergyData."""
    api = AutoCalcDistribution(
        pot=path_file + "mlps/polymlp.lammps.pair.Ag",
        path_output="tmp",
    )
    vaspruns = [
        path_file + "others/vasprun-00001-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
    ]
    energy_data = api._eval_energies(vaspruns)

    np.testing.assert_equal(energy_data.n_atom, [64, 64])
    assert energy_data.names is None
    assert len(energy_data.structures) == 2
    np.testing.assert_allclose(energy_data.energies_mlp, [-62.31644036, -71.33770968])
    np.testing.assert_allclose(energy_data.energies_dft, [-62.57012686, -71.28569153])
    np.testing.assert_allclose(
        energy_data.energies_mlp_per_atom,
        [-0.97369438, -1.11465171],
    )
    np.testing.assert_allclose(
        energy_data.energies_dft_per_atom,
        [-0.97765823, -1.11383893],
    )

    energy_data.names = api._set_structure_names(vaspruns)
    data = energy_data.get_comparison_data()

    assert data.shape == (2, 3)
    shutil.rmtree("tmp")


def test_FormationEnergyData():
    """Test FormationEnergyData."""
    vaspruns = [
        path_file + "others/vasprun-00001-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
        path_file + "others/vasprun-end-Ti.xml",
        path_file + "others/vasprun-end-Al.xml",
    ]
    icsd_ids = ["99787-01", "99787-10", "99787-10", "652876", "52914"]

    api = AutoCalcDistribution(
        pot=path_file + "mlps/polymlp.lammps.gtinv.Ti-Al",
        path_output="tmp",
    )
    api.calc_formation_energies(
        vaspruns=vaspruns,
        icsd_ids=icsd_ids,
        geometry_optimization=False,
    )
    formation_data = api._formation_energies
    formation_data.save_data("tmp/tmp.yaml")
    shutil.rmtree("tmp")
