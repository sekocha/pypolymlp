"""Tests of command line calculations"""

import glob
import os
import shutil
import subprocess
from pathlib import Path

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"
common_str = " --pot " + pot + " --poscar " + poscar + " "


def test_command_lines_calc_prop():
    """Test command lines."""
    cmd = "pypolymlp-calc --properties --eos --elastic" + common_str
    subprocess.run(cmd.split(), check=True)

    for target in glob.glob("polymlp_*"):
        os.remove(target)


def test_command_lines_phonon():
    """Test command lines."""
    cmd = "pypolymlp-calc --phonon --supercell 1 1 1" + common_str
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-calc --phonon --supercell 1 0 0 0 1 0 1 0 2" + common_str
    subprocess.run(cmd.split(), check=True)

    for target in glob.glob("phonon_*"):
        os.remove(target)
    os.remove("polymlp_phonon.yaml")
    os.remove("fc2.hdf5")
    shutil.rmtree("polymlp_phonon_qha")


def test_command_lines_fc():
    """Test command lines."""
    cmd = (
        "pypolymlp-calc --force_constants --supercell 2 2 2 --fc_order 2 "
        "--fc_n_samples 100 --batch_size 50 --disp 0.01 --is_plusminus" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-calc --force_constants --supercell 2 0 0 0 2 0 0 0 2 --fc_order 2 "
        "--fc_n_samples 100 --batch_size 50 --disp 0.01 --is_plusminus" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-calc --force_constants --supercell 2 2 2 --fc_order 2 "
        "--use_gradient_solver "
        "--fc_n_samples 100 --batch_size 50 --disp 0.01 --is_plusminus" + common_str
    )
    subprocess.run(cmd.split(), check=True)

    os.remove("fc2.hdf5")


def test_command_lines_go():
    """Test command lines."""
    cmd = (
        "pypolymlp-calc --geometry_optimization --pressure 0.1 "
        "--scale_axis 10 --maxiter 10 --gtol 0.01" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-calc --geometry_optimization --pressure 0.1 --no_symmetry "
        "--method CG --fix_cell "
        "--scale_axis 10 --maxiter 10 --gtol 0.01" + common_str
    )
    subprocess.run(cmd.split(), check=True)

    os.remove("POSCAR_eqm")


def test_command_lines_features():
    """Test command lines."""
    cmd = "pypolymlp-calc --features" + common_str
    subprocess.run(cmd.split(), check=True)

    os.remove("features.npy")


def test_command_lines_gsfe():
    """Test command lines."""
    cmd = (
        "pypolymlp-calc --gsfe --disp1 1 0 0 --disp2 0 1 0 --glide 0 0 1 "
        " --n_layers 2 --n_points 2" + common_str
    )
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-calc --gsfe --disp1 1 0 0 --disp2 0 1 0 --glide 0 0 1 "
        " --n_layers 2 --frac1 0.1 --frac2 0.1" + common_str
    )
    subprocess.run(cmd.split(), check=True)

    shutil.rmtree("poscars")
    os.remove("gsfe.dat")
    os.remove("POSCAR_gsf")
