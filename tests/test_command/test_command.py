"""Tests of command lines"""

import subprocess


def test_command_lines():
    """Test command lines."""

    cmd = "pypolymlp --help"
    subprocess.run(cmd.split(), check=True)

    cmd = "pypolymlp-calc --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-autocalc --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-utils --help"
    subprocess.run(cmd.split(), check=True)

    cmd = "pypolymlp-structure --help"
    subprocess.run(cmd.split(), check=True)

    cmd = "pypolymlp-sscha --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-sscha-post --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-sscha-structure --help"
    subprocess.run(cmd.split(), check=True)

    cmd = "pypolymlp-md --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-thermodynamics --help"
    subprocess.run(cmd.split(), check=True)


def test_command_lines_developer():
    """Test command lines."""
    cmd = "pypolymlp-symfc --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-kim --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-invariant --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-lammps --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-lammps-autocalc --help"
    subprocess.run(cmd.split(), check=True)
