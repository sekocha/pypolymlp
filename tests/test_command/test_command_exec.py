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
    cmd = "pypolymlp-calc --elastic --properties --eos" + common_str
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
