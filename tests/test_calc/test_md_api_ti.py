"""Tests of thermodynamic integration."""

import os
from pathlib import Path

import pytest

from pypolymlp.calculator.md.api_md import PolymlpMD
from pypolymlp.calculator.md.api_ti import PolymlpTI
from pypolymlp.calculator.properties import Properties

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"
fc2hdf5 = path_file + "others/fc2_Al_111.hdf5"
properties = Properties(pot=pot)


def test_thermodynamic_integration():
    """Test thermodynamic integration."""
    md = PolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_fc2(properties=properties, fc2hdf5=fc2hdf5, alpha=0.0)

    ti = PolymlpTI(md, verbose=True)
    ti.run_thermodynamic_integration(
        n_alphas=3,
        max_alpha=0.99,
        temperature=700,
        n_eq=2,
        n_steps=3,
    )
    assert ti.alpha == pytest.approx(0.99)
    assert isinstance(ti.free_energy, float)
    assert isinstance(ti.energy, float)
    assert isinstance(ti.entropy, float)
    assert len(ti.logs) == 5

    ti.save_ti_yaml("tmp_ti.yaml")
    os.remove("tmp_ti.yaml")

    for log in ti.logs:
        assert log.alpha is not None
        assert log.average_energy is not None
        assert log.average_total_energy is not None
        assert log.average_displacement is not None
        assert log.average_energy_from_alpha0 is not None
        assert log.free_energy_perturb is not None
        assert log.free_energy_perturb_order1 is not None


# def test_thermodynamic_integration():
#    """Test thermodynamic integration."""
#    md = run_thermodynamic_integration(
#        pot=pot,
#        pot_ref=None,
#        poscar=poscar,
#        supercell_size=(1, 1, 1),
#        fc2hdf5=fc2hdf5,
#        n_alphas=3,
#        max_alpha=0.99,
#        temperature=700,
#        n_eq=2,
#        n_steps=3,
#        heat_capacity=False,
#        filename="tmp_ti.yaml",
#        verbose=True,
#    )
#    os.remove("tmp_ti.yaml")
#    assert md.total_free_energy is not None
#    assert md.total_free_energy_order1 is not None
#    assert md.reference_free_energy is not None
#    assert md.free_energy is not None
#    assert md.free_energy_order1 is None
#    assert md.delta_heat_capacity is None


# def test_thermodynamic_integration_use_ref():
#     """Test thermodynamic integration."""
#     md = run_thermodynamic_integration(
#         pot=pot,
#         pot_ref=pot,
#         poscar=poscar,
#         supercell_size=(1, 1, 1),
#         fc2hdf5=fc2hdf5,
#         n_alphas=3,
#         max_alpha=1.0,
#         temperature=700,
#         n_eq=2,
#         n_steps=3,
#         heat_capacity=False,
#         filename="tmp_ti.yaml",
#         verbose=True,
#     )
#     os.remove("tmp_ti.yaml")
#     assert md.total_free_energy is not None
#     assert md.total_free_energy_order1 is not None
#     assert md.reference_free_energy is not None
#     assert md.free_energy is not None
#     assert md.free_energy_order1 is not None
#     assert md.delta_heat_capacity is None
#
#     assert md.total_free_energy == pytest.approx(md.total_free_energy_order1)
#     assert md.free_energy == pytest.approx(0.0)
#     assert md.free_energy_order1 == pytest.approx(0.0)
