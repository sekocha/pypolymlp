"""Tests of command lines"""

import subprocess


def test_command_lines():
    """Test command lines."""

    cmd = "pypolymlp --help"
    subprocess.call(cmd.split())
    cmd = "pypolymlp-calc --help"
    subprocess.call(cmd.split())
    cmd = "pypolymlp-sscha --help"
    subprocess.call(cmd.split())
    cmd = "pypolymlp-sscha-post --help"
    subprocess.call(cmd.split())
    cmd = "pypolymlp-sscha-structure --help"
    subprocess.call(cmd.split())
    cmd = "pypolymlp-structure --help"
    subprocess.call(cmd.split())
    cmd = "pypolymlp-utils --help"
    subprocess.call(cmd.split())
