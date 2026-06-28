"""Classes and functions used for setting structure in lammps format."""

import numpy as np
from numpy.typing import NDArray

# from pypolymlp.core.data_format import PolymlpStructure


def triangularize_axis(axis: NDArray):
    """Triangularize axis.

    Return
    ------
    tri_axis: Non-zero axis six elements (xx, yy, zz, xy, yz, zx).
    tri_axis_full: Triangularized axis matrix.
    rotation: Rotation matrix for triangularization. TriA = R @ A.
    """
    a, b, c = np.linalg.norm(axis, axis=0)
    calpha = (axis[:, 1] @ axis[:, 2]) / (b * c)
    cbeta = (axis[:, 2] @ axis[:, 0]) / (c * a)
    cgamma = (axis[:, 0] @ axis[:, 1]) / (a * b)

    lx = a
    xy = b * cgamma
    xz = c * cbeta
    ly = np.sqrt(b * b - xy * xy)
    yz = (b * c * calpha - xy * xz) / ly
    lz = np.sqrt(c * c - xz * xz - yz * yz)

    tri_axis = np.array([lx, ly, lz, xy, yz, xz])
    tri_axis_full = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])
    rotation = tri_axis_full @ np.linalg.inv(axis)
    return (tri_axis, tri_axis_full, rotation)


#    def to_initial_basis(self, lmp_cartesian: np.ndarray):
#        """Return fractional coordinates in initial basis."""
#        if self.rotation_inverse is None:
#            raise ValueError("No definition of inverse rotation.")
#        return self.rotation_inverse @ lmp_cartesian
