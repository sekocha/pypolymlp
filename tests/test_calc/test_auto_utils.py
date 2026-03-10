"""Tests of utility functions for calculating properties automatically."""

import os
from pathlib import Path

import numpy as np
import pytest

from pypolymlp.calculator.auto.autocalc_utils import AutoCalcBase, Prototype

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_AutoCalcBase():
    """Test AutoCalcBase."""
    pot = path_file + "mlps/polymlp.lammps.pair.Ag"
    base = AutoCalcBase(pot=pot)
    assert base.pot == pot
    assert base.properties is not None
    assert base.calc_api is not None
    assert tuple(base.element_strings) == tuple(["Ag"])
    assert base.n_types == 1
    assert base.path_output == "."
    assert base.path_header == "./polymlp_"


def test_prototype_class(structure_rocksalt):
    """Test Prototype class."""
    prot = Prototype(
        structure=structure_rocksalt,
        name="RS",
        icsd_id="00001",
        n_atom=8,
        phonon_supercell=(2, 2, 2),
    )
    assert prot.lattice_constants is None
    assert prot.lattice_constants_dft is None
    assert prot.volume_dft is None

    prot.structure_eq = structure_rocksalt
    prot.structure_dft = structure_rocksalt
    values = [4.0, 4.0, 4.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(prot.lattice_constants, values)
    np.testing.assert_allclose(prot.lattice_constants_dft, values)
    np.testing.assert_allclose(prot.volume_dft, 64)

    prot.elastic_constants = np.random.random((6, 6))

    data = np.random.random((5, 2))
    prot.set_eos_data(e0=-3.5, v0=10.0, b0=50.0, eos_mlp=data, eos_fit=data)
    prot.set_qha_data(
        temperatures=[0, 100, 200],
        thermal_expansion=[1e-5, 2e-5, 3e-5],
        bulk_modulus=[50, 49, 48],
    )
    prot.save_properties(filename="tmp.yaml")
    os.remove("tmp.yaml")

    is_element = prot.is_element("Mg")
    assert is_element == False

    comp = prot.get_composition(element_strings=("Mg", "O"))
    assert comp == pytest.approx(0.5)
