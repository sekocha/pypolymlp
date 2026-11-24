"""Functions for validation."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_core_utils import refine_positions, sort_wrt_types
from pypolymlp.utils.supercell_utils import get_supercell


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

    def check_rotation(self, etol: float = 1e-12, ftol: float = 1e-12):
        """Check rotational invariance."""
        R = Rotation.random().as_matrix()
        st = copy.deepcopy(self._structure)
        st.axis = R @ st.axis

        e, f, s = self._prop.eval(st)
        assert e == pytest.approx(self._e, abs=etol)
        np.testing.assert_allclose(f, R @ self._f, atol=ftol)

    def check_translation(
        self, etol: float = 1e-12, ftol: float = 1e-12, stol: float = 1e-12
    ):
        """Check translational invariance."""
        st = copy.deepcopy(self._structure)
        st.positions = refine_positions(positions=st.positions + 0.1)

        e, f, s = self._prop.eval(st)
        self._assert_all(e, f, s, etol=etol, ftol=ftol, stol=stol)

    def check_permute_axis(
        self, etol: float = 1e-12, ftol: float = 1e-12, stol: float = 1e-12
    ):
        """Check invariance for axis permutation."""
        st = copy.deepcopy(self._structure)
        order = np.array([2, 0, 1])
        st.axis = st.axis[:, order]
        st.positions = st.positions[order]

        e, f, s = self._prop.eval(st)
        self._assert_all(e, f, s, etol=etol, ftol=ftol, stol=stol)

    def check_permute_atom(
        self, etol: float = 1e-12, ftol: float = 1e-12, stol: float = 1e-12
    ):
        """Check invariance for atom permutation.

        Check structures such as those shown below.
        types = [0,1,1,0]
        elements = ["Mg","O","O","Mg"]
        """
        st = copy.deepcopy(self._structure)
        order = np.random.permutation(len(self._structure.elements))

        st.positions = st.positions[:, order]
        st.types = st.types[order]
        st.elements = st.elements[order]

        e, f, s = self._prop.eval(st)
        assert e == pytest.approx(self._e, abs=etol)
        np.testing.assert_allclose(f, self._f[:, order], atol=ftol)
        np.testing.assert_allclose(s, self._s, atol=stol)

        st, idx = sort_wrt_types(st, return_ids=True)

        e, f, s = self._prop.eval(st)
        assert e == pytest.approx(self._e, abs=etol)
        np.testing.assert_allclose(f, (self._f[:, order[idx]]), atol=ftol)
        np.testing.assert_allclose(s, self._s, atol=stol)

    def check_different_types_representation(
        self, etol: float = 1e-12, ftol: float = 1e-12, stol: float = 1e-12
    ):
        """Check invariance for structures with different type representation.

        Check structures such as those shown below.
        types = [1,1,0,0]
        elements = ["Mg","Mg","O","O"]
        """
        st = copy.deepcopy(self._structure)
        types = copy.deepcopy(self._structure.types)
        order = np.random.permutation(len(self._structure.n_atoms))
        for i, replace in enumerate(order):
            self._structure.types[types == i] = replace

        e, f, s = self._prop.eval(st)
        self._assert_all(e, f, s, etol=etol, ftol=ftol, stol=stol)

    def check_different_primitive(self, etol: float = 1e-12):
        """Check invariance for different primitive cells."""
        # TODO: Expand more systematically
        expansions = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
            [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[1, 0, 0], [-1, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [-1, 1, 0], [-1, -1, 1]],
            [[1, 0, 0], [-3, 1, 0], [3, 3, 1]],
            [[1, 0, 0], [3, 1, 0], [3, 3, 1]],
            [[1, 0, 0], [-3, 1, 0], [-3, -3, 1]],
            [[1, 0, 0], [5, 1, 0], [5, 5, 1]],
            [[1, 0, 0], [-5, 1, 0], [-5, -5, 1]],
            [[1, 0, 0], [10, 1, 0], [10, 10, 1]],
            [[1, 0, 0], [-10, 1, 0], [-10, -10, 1]],
            [[1, 3, 3], [0, 1, 3], [0, 0, 1]],
            [[1, 4, 4], [0, 1, 4], [0, 0, 1]],
            [[1, 5, 5], [0, 1, 5], [0, 0, 1]],
            [[1, -5, -5], [0, 1, -5], [0, 0, 1]],
            [[1, 10, 10], [0, 1, 10], [0, 0, 1]],
            [[1, -10, -10], [0, 1, -10], [0, 0, 1]],
            [[1, 0, 0], [5, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [-5, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [10, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [-10, 1, 0], [0, 0, 1]],
        ]
        structures = [
            get_supercell(self._structure, np.array(hnf)) for hnf in expansions
        ]
        e, _, _ = self._prop.eval_multiple(structures)
        np.testing.assert_allclose(e, self._e, atol=etol)

    def check_supercell(self, etol: float = 1e-12):
        """Check invariance for supercell expansion."""
        expansions = [
            [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[2, 0, 0], [1, 2, 0], [1, 1, 2]],
            [[1, 0, 0], [4, 3, 0], [0, 0, 3]],
        ]
        structures = [
            get_supercell(self._structure, np.array(hnf)) for hnf in expansions
        ]
        sizes = np.array([np.linalg.det(hnf) for hnf in expansions])
        e, _, _ = self._prop.eval_multiple(structures)
        np.testing.assert_allclose(e / sizes, self._e, atol=etol)
