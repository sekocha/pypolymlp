"""Tests of command line utilities"""

import os
import subprocess
from pathlib import Path

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"
common_str = " --pot " + pot + " --poscar " + poscar + " "


def test_vasprun_compress():
    """Test command lines."""
    cmd = "pypolymlp-utils --vasprun_compress"
    subprocess.run(cmd.split(), check=True)


def test_atomic_energy():
    """Test command lines."""
    cmd = "pypolymlp-utils --atomic_energy_elements Mg O"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-utils --atomic_energy_formula MgO"
    subprocess.run(cmd.split(), check=True)


def test_benchmark():
    """Test command lines."""
    cmd = "pypolymlp-utils --benchmark --test"
    subprocess.run(cmd.split(), check=True)


def test_calc_cost():
    """Test command lines."""
    cmd = "pypolymlp-utils --calc_cost" + common_str
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-utils --calc_cost --supercell 2 0 0 0 2 0 1 0 2" + common_str
    subprocess.run(cmd.split(), check=True)

    os.remove("polymlp_cost.yaml")


def test_spglib():
    """Test command lines."""
    cmd = "pypolymlp-utils --refine_cell --symprec 1e-3 --space_group" + common_str
    subprocess.run(cmd.split(), check=True)

    os.remove("poscar_pypolymlp")


def test_supercell():
    """Test command lines."""
    cmd = "pypolymlp-utils --supercell 2 2 2" + common_str
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-utils --supercell 2 0 0 0 2 0 0 0 2" + common_str
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-utils --supercell 2 0 0 0 2 0 1 0 2" + common_str
    subprocess.run(cmd.split(), check=True)

    os.remove("sposcar_pypolymlp")


# def test_yaml():
#     """Test command lines."""
#     cmd = "pypolymlp-utils --yaml_converter"
#     subprocess.run(cmd.split(), check=True)
