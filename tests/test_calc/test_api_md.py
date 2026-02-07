"""Tests of MD calculations using API."""

import os
from pathlib import Path

from pypolymlp.api.pypolymlp_md import PypolymlpMD, run_thermodynamic_integration

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"
fc2hdf5 = path_file + "others/fc2_Al_111.hdf5"


def test_md_standard1():
    """Test MD calculations from polymlp using API."""
    md = PypolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator(pot=pot)
    md.run_md_nvt(
        thermostat="Nose-Hoover",
        temperature=700,
        n_eq=2,
        n_steps=3,
        interval_save_forces=1,
        interval_save_trajectory=1,
        interval_log=1,
        logfile="tmp.dat",
    )
    md.save_yaml(filename="tmp.yaml")
    os.remove("tmp.dat")
    os.remove("tmp.yaml")

    assert md.unitcell is not None
    assert md.supercell is not None
    assert md.calculator is not None
    assert md.alpha is None
    assert len(md.energies) == 6
    assert md.forces.shape == (6, 4, 3)
    assert len(md.trajectory) == 6
    assert md.average_energy is not None
    assert md.average_total_energy is not None
    assert md.heat_capacity is not None
    assert md.average_displacement is None
    assert md.delta_energies_10 is None
    assert md.delta_energies_1a is None
    assert md.average_delta_energy_10 is None
    assert md.average_delta_energy_1a is None
    assert md.free_energy_perturb is None
    assert md.total_free_energy is None
    assert md.total_free_energy_order1 is None
    assert md.reference_free_energy is None
    assert md.free_energy is None
    assert md.free_energy_order1 is None
    assert md.delta_heat_capacity is None
    assert md.final_structure is not None


def test_md_standard2():
    """Test MD calculations from polymlp using API."""
    md = PypolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator(pot=pot)
    md.run_md_nvt(
        thermostat="Langevin",
        temperature=700,
        n_eq=2,
        n_steps=3,
        interval_save_forces=1,
        interval_save_trajectory=1,
        interval_log=1,
        logfile="tmp.dat",
    )
    md.save_yaml(filename="tmp.yaml")
    os.remove("tmp.dat")
    os.remove("tmp.yaml")


def test_md_use_fc2_ref():
    """Test MD calculations from polymlp using API."""
    md = PypolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_fc2(pot=pot, fc2hdf5=fc2hdf5, alpha=0.5)
    md.run_md_nvt(
        thermostat="Langevin",
        temperature=700,
        n_eq=2,
        n_steps=3,
        interval_save_forces=1,
        interval_save_trajectory=1,
        interval_log=1,
        logfile="tmp.dat",
    )
    md.save_yaml(filename="tmp.yaml")
    os.remove("tmp.dat")
    os.remove("tmp.yaml")

    assert md.unitcell is not None
    assert md.supercell is not None
    assert md.calculator is not None
    assert md.alpha == 0.5
    assert len(md.energies) == 6
    assert md.forces.shape == (6, 4, 3)
    assert len(md.trajectory) == 6
    assert md.average_energy is not None
    assert md.average_total_energy is not None
    assert md.heat_capacity is not None
    assert md.average_displacement is not None
    assert len(md.delta_energies_10) == 6
    assert len(md.delta_energies_1a) == 6
    assert md.average_delta_energy_10 is not None
    assert md.average_delta_energy_1a is not None
    assert md.free_energy_perturb is not None

    assert md.total_free_energy is None
    assert md.total_free_energy_order1 is None
    assert md.reference_free_energy is None
    assert md.free_energy is None
    assert md.free_energy_order1 is None
    assert md.delta_heat_capacity is None
    assert md.final_structure is not None


def test_md_use_polymlp_ref():
    """Test MD calculations from polymlp using API."""
    md = PypolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_reference(pot=pot, pot_ref=pot, alpha=0.5)
    md.run_md_nvt(
        thermostat="Langevin",
        temperature=700,
        n_eq=2,
        n_steps=3,
        interval_save_forces=1,
        interval_save_trajectory=1,
        interval_log=1,
        logfile="tmp.dat",
    )
    md.save_yaml(filename="tmp.yaml")
    os.remove("tmp.dat")
    os.remove("tmp.yaml")


def test_md_use_general_ref():
    """Test MD calculations from polymlp using API."""
    md = PypolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_general_reference(
        pot_final=pot,
        pot_ref=pot,
        fc2hdf5=fc2hdf5,
        alpha=0.5,
        alpha_final=0.2,
        alpha_ref=0.3,
    )
    md.run_md_nvt(
        thermostat="Langevin",
        temperature=700,
        n_eq=2,
        n_steps=3,
        interval_save_forces=1,
        interval_save_trajectory=1,
        interval_log=1,
        logfile="tmp.dat",
    )
    md.save_yaml(filename="tmp.yaml")
    os.remove("tmp.dat")
    os.remove("tmp.yaml")


def test_thermodynamic_integration():
    """Test thermodynamic integration."""
    md = run_thermodynamic_integration(
        pot=pot,
        pot_ref=None,
        poscar=poscar,
        supercell_size=(1, 1, 1),
        fc2hdf5=fc2hdf5,
        n_alphas=3,
        max_alpha=0.99,
        temperature=700,
        n_eq=2,
        n_steps=3,
        heat_capacity=False,
        filename="tmp_ti.yaml",
        verbose=True,
    )
    os.remove("tmp_ti.yaml")
    assert md.total_free_energy is not None
    assert md.total_free_energy_order1 is not None
