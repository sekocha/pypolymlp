"""Functions for validation."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_core_utils import refine_positions


class PolymlpValidation:
    """Class for validation."""

    def __init__(
        self,
        st: PolymlpStructure,
        properties: Properties,
    ):
        """Init method."""
        self._structure = st
        self._prop = properties
        self._e, self._f, self._s = self._prop.eval(st)

    def _assert_all(
        self,
        e: float,
        f: np.ndarray,
        s: np.ndarray,
        etol: float = 1e-12,
        ftol: float = 1e-12,
        stol: float = 1e-12,
    ):
        """Assert properties."""
        assert e == pytest.approx(self._e, abs=etol)
        np.testing.assert_allclose(f, self._f, atol=ftol)
        np.testing.assert_allclose(s, self._s, atol=stol)

    def check_rotation(self, etol: float = 1e-12):
        """Check rotational invariance."""
        R = Rotation.random().as_matrix()
        st_rot = copy.deepcopy(self._structure)
        st_rot.axis = R @ st_rot.axis
        e, f, s = self._prop.eval(st_rot)
        assert e == pytest.approx(self._e, abs=etol)

    def check_translation(
        self, etol: float = 1e-12, ftol: float = 1e-12, stol: float = 1e-12
    ):
        """Check translational invariance."""
        st = copy.deepcopy(self._structure)
        st.positions = refine_positions(positions=st.positions + 0.1)

        e, f, s = self._prop.eval(st)
        self._assert_all(e, f, s, etol=etol, ftol=ftol, stol=stol)
