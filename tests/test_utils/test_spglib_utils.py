"""Tests of spglib utility functions."""

from pathlib import Path

import numpy as np
import scipy

from pypolymlp.utils.spglib_utils import SymCell, standardize_cell

cwd = Path(__file__).parent


def test_cell(structure_rocksalt):
    """Test structure converter."""
    sym = SymCell(st=structure_rocksalt)
    refined = sym.refine_cell()
    np.testing.assert_allclose(
        refined.axis, [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    )

    spg = sym.get_spacegroup()
    spgs = sym.get_spacegroup_multiple_prec()
    assert spg == "Fm-3m (225)"
    assert spgs == ["Fm-3m (225)", "Fm-3m (225)", "Fm-3m (225)", "Fm-3m (225)"]


def test_standardize_cell(structure_rocksalt):
    """Test standardize_cell."""
    st = standardize_cell(structure_rocksalt)
    dist = scipy.spatial.distance.cdist(st.positions.T, structure_rocksalt.positions.T)
    ids = np.where(dist < 1e-10)[1]
    np.testing.assert_equal(ids, [0, 1, 2, 3, 6, 7, 4, 5])
    np.testing.assert_equal(st.types, np.repeat([0, 1], 4))
    np.testing.assert_equal(st.elements, np.repeat(["Mg", "O"], 4))
