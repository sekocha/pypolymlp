"""Utility functions for MD."""

# from typing import Optional
#
# import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.core.data_format import PolymlpStructure


def initialize_velocity(structure: PolymlpStructure, md_params: MDParams):
    """Generate initial velocity."""
    if md_params.ensemble == "nvt":
        pass
    elif md_params.ensemble == "nve":
        pass
