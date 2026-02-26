"""Tests of API for calculating properties automatically."""

import shutil
from pathlib import Path

from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_api_autocalc1():
    """Test PypolymlpAutoCalc."""
    api = PypolymlpAutoCalc(
        pot=path_file + "mlps/polymlp.lammps.pair.Ag",
        path_output="tmp",
        verbose=True,
    )
    assert api._element_strings == ["Ag"]
    assert api._n_types == 1

    assert api.prototypes is None
    api.load_structures()
    assert len(api.prototypes) == 18

    api.prototypes = api.prototypes[0:2]
    api.run()
    api.save_properties()

    vaspruns = [
        path_file + "others/vasprun-00001-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
    ]
    api.calc_energy_distribution(
        vaspruns_train=vaspruns,
        vaspruns_test=vaspruns,
        functional="PBE",
    )
    assert api._distribution_train.shape == (2, 2)
    assert api._distribution_test.shape == (2, 2)
    api.plot_energy_distribution(system="Ag", pot_id="pair-00001")

    api.compare_with_dft(
        vaspruns=vaspruns,
        icsd_ids=[104296, 105489],
        functional="PBE",
    )
    api.plot_comparison_with_dft(system="Ag", pot_id="pair-00001")
    shutil.rmtree("tmp")


def test_api_autocalc2():
    """Test PypolymlpAutoCalc."""
    api = PypolymlpAutoCalc(
        pot=path_file + "mlps/polymlp.lammps.gtinv.Ti-Al",
        path_output="tmp",
        verbose=True,
    )
    assert api._element_strings == ["Ti", "Al"]
    assert api._n_types == 2

    assert api.prototypes is None
    api.load_structures()
    assert len(api.prototypes) == 39

    api.prototypes = api.prototypes[0:2]
    api.run()
    api.save_properties()

    vaspruns = [
        path_file + "others/vasprun-00001-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
    ]
    api.calc_energy_distribution(
        vaspruns_train=vaspruns,
        vaspruns_test=vaspruns,
        functional="PBE",
    )
    assert api._distribution_train.shape == (2, 2)
    assert api._distribution_test.shape == (2, 2)
    api.plot_energy_distribution(system="Ti-Al", pot_id="polymlp-00001")

    api.compare_with_dft(
        vaspruns=vaspruns,
        icsd_ids=["99787-01", "99787-10"],
        functional="PBE",
    )
    api.plot_comparison_with_dft(system="Ti-Al", pot_id="polymlp-00001")
    shutil.rmtree("tmp")
