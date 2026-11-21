"""Tests of command line calculations"""

import os
import subprocess
from pathlib import Path

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"

poscar = path_file + "POSCAR.fcc.Ag"
pot = path_file + "polymlp.yaml.pair.Ag"
common_str = " --pot " + pot + " --poscar " + poscar + " "


def test_md():
    """Test command lines."""
    cmd = (
        "pypolymlp-md --supercell 1 1 1 --temp 100 "
        "--n_eq 5 --n_steps 10 --output_path ." + common_str
    )
    subprocess.run(cmd.split(), check=True)
    cmd = (
        "pypolymlp-md --supercell 1 0 0 0 1 0 0 0 1 --temp 100 "
        "--n_eq 5 --n_steps 10 --output_path ." + common_str
    )
    subprocess.run(cmd.split(), check=True)

    os.remove("polymlp_md.yaml")
    os.remove("polymlp_md_log.dat")
