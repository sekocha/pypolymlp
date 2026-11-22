"""Tests of command line utilities"""

import glob
import os
import shutil
import subprocess
from pathlib import Path

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"
common_pot = " --pot " + pot
common_poscar = " --poscar " + poscar
common_str = " --pot " + pot + " --poscar " + poscar + " "


# def test_autocalc():
#     """Test command lines."""
#     cmd = "pypolymlp-autocalc" + common_pot
#     subprocess.run(cmd.split(), check=True)


def test_invariant():
    """Test command lines."""
    cmd = "pypolymlp-invariant --orders 3 --minl 2 --maxl 4"
    subprocess.run(cmd.split(), check=True)
    for f in glob.glob("polyinv_*"):
        os.remove(f)


def test_kim():
    """Test command lines."""
    cmd = "pypolymlp-kim" + common_pot
    subprocess.run(cmd.split(), check=True)
    for f in glob.glob("Polymlp_Seko_2026p1_Ag_*"):
        shutil.rmtree(f)


def test_structure():
    """Test command lines."""
    poscar_cmd = " --poscars " + poscar
    cmd = "pypolymlp-structure --displacements 2 --const_distance 0.1" + poscar_cmd
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-structure --displacements 2 --max_distance 0.1" + poscar_cmd
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-structure --displacements 2 --supercell 2 2 2 "
        "--const_distance 0.1" + poscar_cmd
    )
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-structure --isotropic 2 "
        "--n_volumes 3 --min_volume 10 --max_volume 20" + poscar_cmd
    )
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-structure --standard 3 --max_distance 0.1" + poscar_cmd
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-structure --low_density 3 --max_distance 0.1" + poscar_cmd
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-structure --high_density 3 --max_distance 0.1" + poscar_cmd
    subprocess.run(cmd.split(), check=True)

    os.remove("polymlp_str_samples.yaml")
    shutil.rmtree("poscars")


def test_sscha_post():
    """Test command lines."""
    fc2 = path_file + "fc2.hdf5"
    yaml = path_file + "sscha_results.yaml"
    cmd = "pypolymlp-sscha-post --distribution -n 10 --yaml " + yaml + " --fc2 " + fc2
    subprocess.run(cmd.split(), check=True)

    for f in glob.glob("sscha_*"):
        os.remove(f)


def test_symfc():
    """Test command lines."""
    cmd1 = " --poscar " + path_file + "POSCAR-MgO"
    cmd2 = (
        " --vaspruns "
        + path_file
        + "vasprun-00001-MgO.xml "
        + path_file
        + "vasprun-00002-MgO.xml"
    )
    cmd = (
        "pypolymlp-symfc --orders 2 3 --supercell 2 2 2 --batch_size 10 "
        "--cutoff_fc2 4 --cutoff_fc3 4" + cmd1 + cmd2
    )
    subprocess.run(cmd.split(), check=True)
    os.remove("fc2.hdf5")
    os.remove("fc3.hdf5")
    os.remove("kappa-m191919.hdf5")
