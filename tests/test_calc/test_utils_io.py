"""Tests of utility functions for IO."""

import os

from pypolymlp.calculator.utils.io_utils import print_pot


def test_print_pot():
    """Test print_pot."""
    pot = "polymlp.yaml"
    with open("tmp.yaml", "w") as f:
        print_pot(pot, tag="polymlp", indent=0, file=f)

    pot = ["polymlp.yaml", "polymlp2.yaml"]
    with open("tmp.yaml", "w") as f:
        print_pot(pot, tag="polymlp", indent=0, file=f)
    os.remove("tmp.yaml")
