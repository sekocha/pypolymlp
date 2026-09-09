"""Class for calculating transformation path."""

import copy

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.supercell_utils import get_supercell

# from pypolymlp.utils.vasp_utils import write_poscar_file


class PolymlpTransformation:
    """Class for calculating transformation pathways."""

    def __init__(
        self,
        structure: PolymlpStructure,
        properties: Properties,
        verbose: bool = False,
    ):
        """Init method."""
        self._prop = properties
        self._verbose = verbose

        self._base_structure = structure
        self._supercell = structure
        self._supercell_rev = None

        self._axis1, self._axis2, self._other = None, None, None
        self._sd_cell = None
        self._sd_pos = None

    def set_supercell(self, supercell_matrix: np.ndarray):
        """Set supercell."""
        if np.array(supercell_matrix).shape != (3, 3):
            raise RuntimeError("Supercell matrix shape is not (3, 3).")

        for i in range(3):
            if np.array(supercell_matrix)[i, i] < 0:
                raise RuntimeError("Found negative diagonal elements of matrix")

        self._supercell = get_supercell(self._base_structure, supercell_matrix)
        return self._supercell

    def change_angle(self, degs: float, axis1: int = 0, axis2: int = 1):
        """Change angle between two axes.

        axis2 is rotated while axis1 and the other axis is fixed.

        [a1, a2, a3].T @ new_a2 = [|a1||a2|cos(degs), a2 @ a2, a2 @ a3].T
        """
        other = ({0, 1, 2} - {axis1, axis2}).pop()
        degs_rad12 = degs * np.pi / 180

        st = self._supercell
        a1, a2, a3 = st.axis[:, axis1], st.axis[:, axis2], st.axis[:, other]
        len1, len2 = np.linalg.norm(a1), np.linalg.norm(a2)
        prod21 = len1 * len2 * np.cos(degs_rad12)
        prod22 = len2 * len2
        prod2other = a2 @ a3
        val = [prod21, prod22, prod2other]

        axisT = np.vstack([a1, a2, a3])
        new_axis = np.linalg.inv(axisT) @ val

        self._supercell_rev = copy.deepcopy(self._supercell)
        self._supercell_rev.axis[:, axis2] = new_axis
        self._axis1, self._axis2, self._other = axis1, axis2, other
        return self._supercell_rev

    def _change_basis(self):
        """Change basis for angle-fixed optimization."""
        basis_a = self._geometry._basis.basis_a
        basis_a_new = np.zeros((basis_a.shape[0], basis_a.shape[1] + 2))
        basis_a_new[:, : basis_a.shape[1]] = basis_a

        axis = self._supercell_rev.axis
        vec = np.zeros((3, 3))
        vec[self._axis1, self._axis1] = 1
        basis_a_new[:, basis_a.shape[1]] = (axis @ vec).reshape(-1)

        vec = np.zeros((3, 3))
        vec[self._other, self._other] = 1
        basis_a_new[:, basis_a.shape[1] + 1] = (axis @ vec).reshape(-1)

        basis_a_new, _ = np.linalg.qr(basis_a_new)
        self._geometry.change_basis_axis(basis_a_new)
        return self._geometry

    def run_single_fix_angle(self, gtol: float = 1e-4):
        """Run a single geometry optimization with fixed angles."""
        if self._supercell_rev is None:
            raise RuntimeError("Structure for calculation not found.")

        self._geometry = GeometryOptimization(
            self._supercell_rev,
            self._prop,
            with_sym=True,
            relax_cell=False,
            relax_volume=True,
            relax_positions=True,
            verbose=self._verbose,
        )
        self._change_basis()
        self._geometry.run(gtol=gtol)
        return self._geometry

    @property
    def energy(self):
        """Return energy."""
        return self._geometry.energy

    @property
    def success(self):
        """Return whether optimization is successed."""
        return self._geometry.success

    @property
    def converged_structure(self):
        """Return whether optimization is successed."""
        return self._geometry.structure
