"""Functions for constructing projector."""

import numpy as np

from pypolymlp.polyinv.cxx.lib import libprojcpp
from pypolymlp.polyinv.polyinv_utils import matrix_index_to_lm


def build_projector(lcomb: list):
    """Build projector for l values.

    Reference: Quantum theory of angular momentum (Varshalovich, p.96)
    """
    obj = libprojcpp.Projector()
    obj.build_projector(lcomb)
    core, row = obj.get_core(), obj.get_row()
    lm_indices = np.array([matrix_index_to_lm(i, lcomb) for i in row])
    return (core, lm_indices)
