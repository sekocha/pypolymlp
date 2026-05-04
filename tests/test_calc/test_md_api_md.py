"""Tests of MD calculations using API."""

import os
from pathlib import Path

from pypolymlp.calculator.md.api_md import PolymlpMD
from pypolymlp.calculator.properties import Properties

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "poscars/POSCAR.fcc.Al"
pot = path_file + "mlps/polymlp.yaml.gtinv.Al"
fc2hdf5 = path_file + "others/fc2_Al_111.hdf5"
properties = Properties(pot=pot)


def test_util_functions():
    """Test utility functions in PypolymlpMD."""
    md = PolymlpMD(verbose=True)
    fc2file = md.find_reference(path_file + "others", 1000)
    assert fc2file.split("/")[-1] == "fc2.hdf5"


def test_md_standard1():
    """Test MD calculations from polymlp using API."""
    md = PolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator(properties)
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
    assert md.final_structure is not None
    assert md.calculator is not None

    assert len(md.energies) == 6
    assert md.forces.shape == (6, 4, 3)
    assert len(md.trajectory) == 6
    assert md.average_energy is not None
    assert md.average_total_energy is not None
    assert md.heat_capacity is not None
    assert md.average_displacement is None
    assert md.free_energy_perturb is None
    assert md.free_energy_perturb_order1 is None

    assert md.delta_energies_10 is None
    assert md.delta_energies_1a is None
    assert md.average_delta_energy_10 is None
    assert md.average_delta_energy_1a is None

    assert md.supercell_matrix.shape == (3, 3)
    assert not md.use_reference


def test_md_standard2():
    """Test MD calculations from polymlp using API."""
    md = PolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator(properties=properties)
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
    md = PolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_fc2(properties=properties, fc2hdf5=fc2hdf5, alpha=0.5)
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
    assert md.final_structure is not None
    assert md.calculator is not None

    assert len(md.energies) == 6
    assert md.forces.shape == (6, 4, 3)
    assert len(md.trajectory) == 6
    assert md.average_energy is not None
    assert md.average_total_energy is not None
    assert md.heat_capacity is not None
    assert md.average_displacement is not None
    assert isinstance(md.free_energy_perturb, float)
    assert isinstance(md.free_energy_perturb_order1, float)

    assert md.delta_energies_10 is not None
    assert md.delta_energies_1a is not None
    assert md.average_delta_energy_10 is not None
    assert md.average_delta_energy_1a is not None

    assert md.supercell_matrix.shape == (3, 3)
    assert md.use_reference
    assert md.fc2file is not None


def test_md_use_polymlp_ref():
    """Test MD calculations from polymlp using API."""
    md = PolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_reference(
        properties=properties, properties_ref=properties, alpha=0.5
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

    assert isinstance(md.free_energy_perturb, float)
    assert isinstance(md.free_energy_perturb_order1, float)

    assert md.delta_energies_10 is not None
    assert md.delta_energies_1a is not None
    assert md.average_delta_energy_10 is not None
    assert md.average_delta_energy_1a is not None

    assert md.supercell_matrix.shape == (3, 3)
    assert md.use_reference
    assert md.fc2file is None


def test_md_use_general_ref():
    """Test MD calculations from polymlp using API."""
    md = PolymlpMD(verbose=True)
    md.load_poscar(poscar)
    md.set_supercell((1, 1, 1))
    md.set_ase_calculator_with_general_reference(
        properties_final=properties,
        properties_ref=properties,
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

    assert isinstance(md.free_energy_perturb, float)
    assert isinstance(md.free_energy_perturb_order1, float)

    assert md.delta_energies_10 is not None
    assert md.delta_energies_1a is not None
    assert md.average_delta_energy_10 is not None
    assert md.average_delta_energy_1a is not None

    assert md.supercell_matrix.shape == (3, 3)
    assert md.use_reference
    assert md.fc2file is not None
