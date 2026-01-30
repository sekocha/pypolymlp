"""Tests of command lines"""

import subprocess


def test_command_lines():
    """Test command lines."""

    cmd = "pypolymlp --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-calc --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-sscha --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-sscha-post --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-sscha-structure --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-structure --help"
    subprocess.run(cmd.split(), check=True)
    cmd = "pypolymlp-utils --help"
    subprocess.run(cmd.split(), check=True)
