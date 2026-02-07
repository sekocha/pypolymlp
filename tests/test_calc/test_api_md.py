"""Tests of MD calculations using API."""

import os
from pathlib import Path

from pypolymlp.api.pypolymlp_md import PypolymlpMD

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
