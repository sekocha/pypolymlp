"""Tests of API for calculating properties automatically."""

import os
import shutil
from pathlib import Path

from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_api_autocalc1():
    """Test PypolymlpAutoCalc."""
    vaspruns = [
        path_file + "others/vasprun-00001-Ag.xml",
        path_file + "others/vasprun-00002-Ag.xml",
    ]

    api = PypolymlpAutoCalc(
        pot=path_file + "mlps/polymlp.lammps.pair.Ag",
        path_output="tmp",
        functional="PBE",
        verbose=False,
    )
    assert api._n_types == 1

    assert api.prototypes is None
    api.load_prototypes()
    assert len(api.prototypes) == 18

    api.prototypes = api.prototypes[0:2]
    api.calc_prototypes()
    api.set_prototypes_from_DFT(vaspruns, ["104296", "105489"])
    api.save_prototypes()

    api.calc_energy_distribution(vaspruns_train=vaspruns, vaspruns_test=vaspruns)
    api.plot_energy_distribution(system="Ag", pot_id="pair-00001")

    api.calc_comparison_with_dft(
        vaspruns=vaspruns * 10,
        icsd_ids=["104296", "105489"] * 10,
    )
    api.plot_comparison_with_dft(system="Ag", pot_id="pair-00001")
    shutil.rmtree("tmp")


def test_api_autocalc2():
    """Test PypolymlpAutoCalc."""
    vaspruns = [
        path_file + "others/vasprun-00001-Ti-Al.xml",
        path_file + "others/vasprun-00002-Ti-Al.xml",
    ]

    api = PypolymlpAutoCalc(
        pot=path_file + "mlps/polymlp.lammps.gtinv.Ti-Al",
        path_output="tmp",
        functional="PBE",
        verbose=True,
    )
    assert api._n_types == 2

    assert api.prototypes is None
    api.load_prototypes()
    assert len(api.prototypes) == 69

    api.prototypes = api.prototypes[0:2]
    api.calc_prototypes()
    api.set_prototypes_from_DFT(vaspruns, ["104296", "105489"])
    api.save_prototypes()

    api.calc_comparison_with_dft(
        vaspruns=vaspruns * 10,
        icsd_ids=["99787-01", "99787-10"] * 10,
    )
    api.plot_comparison_with_dft(system="Ti-Al", pot_id="polymlp-00001")

    vaspruns.append(path_file + "others/vasprun-end-Ti.xml")
    vaspruns.append(path_file + "others/vasprun-end-Al.xml")
    api.calc_formation_energies(vaspruns, ["99787-01", "99787-10", "652876", "52914"])
    api.plot_binary_formation_energies("Ti-Al", "polymlp-00001")

    api.calc_energy_distribution(vaspruns_train=vaspruns, vaspruns_test=vaspruns)
    api.plot_energy_distribution(system="Ti-Al", pot_id="polymlp-00001")

    assert os.path.exists("tmp/polymlp_comparison.dat")
    assert os.path.exists("tmp/polymlp_comparison.png")
    assert os.path.exists("tmp/polymlp_distribution_formation.png")
    assert os.path.exists("tmp/polymlp_distribution.png")
    assert os.path.exists("tmp/polymlp_formation_energy.png")
    assert os.path.exists("tmp/polymlp_formation_energy.yaml")
    assert os.path.exists("tmp/polymlp_size.dat")
    shutil.rmtree("tmp")
