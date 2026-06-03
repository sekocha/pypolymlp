"""Functions for constructing projector."""

import numpy as np
from scipy.sparse import csr_array

from pypolymlp.polyinv.cxx.lib import libprojcpp
from pypolymlp.polyinv.polyinv_utils import matrix_index_to_lm


def build_projector(lcomb: list, mcomb_all: list):
    """Build projector for l values.

    Reference: Quantum theory of angular momentum (Varshalovich, p.96)
    """
    size = np.prod(2 * np.array(lcomb) + 1)
    obj = libprojcpp.Projector()
    obj.build_projector(lcomb, mcomb_all)
    data, row, col = obj.get_data(), obj.get_row(), obj.get_col()

    proj = csr_array((data, (row, col)), shape=(size, size), dtype=float)
    lm_indices = np.array([matrix_index_to_lm(i, lcomb) for i in range(size)])
    return (proj, lm_indices)
