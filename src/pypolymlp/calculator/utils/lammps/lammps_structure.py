"""Classes and functions used for setting structure in lammps format."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def convert_structure_to_lammps_format(structure: PolymlpStructure):
    """Convert structure in PolymlpStructure to structure in LammpsStructure."""
    axis = structure.axis
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

    lmp_axis = np.array([lx, ly, lz, xy, yz, xz])
    lmp_axis_full = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])
    rotation = lmp_axis_full @ np.linalg.inv(axis)

    lmp_st = LammpsStructure(
        axis=lmp_axis,
        types=structure.types,
        elements=structure.elements,
        positions=structure.positions,
        rotation=rotation,
    )
    return lmp_st


@dataclass
class LammpsStructure:
    """Dataclass of structure in lammps_format.

    Parameters
    ----------
    axis: Axis vector [lx, ly, lz, xy, yz, zx], shape=(6).
    types: Atomic type integers, (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
    positions: Scaled positions, shape=(3, n_atom).
    positions_cartesian: Cartesian positions, shape=(3, n_atom).

    (optional)
    elements: Element list, (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
    rotation: Rotation matrix from original structure. lmp_axis = rot @ axis.
    """

    axis: np.ndarray
    types: Union[np.ndarray, list]

    positions: Optional[np.ndarray] = None
    positions_cartesian: Optional[np.ndarray] = None

    elements: Optional[Union[np.ndarray, list]] = None
    n_atomtypes: Optional[int] = None

    axis_matrix: Optional[np.ndarray] = None
    axis_matrix_inverse: Optional[np.ndarray] = None

    rotation: Optional[np.ndarray] = None
    rotation_inverse: Optional[np.ndarray] = None

    verbose: bool = False

    def __post_init__(self):
        """Init method."""
        self.n_atomtypes = len(np.unique(self.types))

        self._axis_check()
        self._position_check()
        self._set_rotation_matrices()

    def _axis_check(self):
        if len(self.axis) != 6:
            raise ValueError("Number of axis elements must be 6.")

        a = self.axis
        self.axis_matrix = np.array([[a[0], a[3], a[5]], [0, a[1], a[4]], [0, 0, a[2]]])
        try:
            self.axis_matrix_inverse = np.linalg.inv(self.axis_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("LammpsStructure: Invalid axis matrix.")

    def _position_check(self):
        if self.positions is None and self.positions_cartesian is None:
            raise ValueError("positions or positions_cartesian required.")

        if self.positions is not None:
            self.positions_cartesian = self.axis_matrix @ np.array(self.positions)
        else:
            self.positions = self.axis_matrix_inverse @ self.positions_cartesian

    def _set_rotation_matrices(self):
        if self.rotation is None:
            self.rotation = np.eye(3)
            self.rotation_inverse = np.eye(3)
        else:
            self.rotation_inverse = np.linalg.inv(self.rotation)

    @property
    def lx(self):
        """Return lx."""
        return self.axis_matrix[0, 0]

    @property
    def ly(self):
        """Return ly."""
        return self.axis_matrix[1, 1]

    @property
    def lz(self):
        """Return lz."""
        return self.axis_matrix[2, 2]

    @property
    def xy(self):
        """Return xy."""
        return self.axis_matrix[0, 1]

    @property
    def yz(self):
        """Return yz."""
        return self.axis_matrix[1, 2]

    @property
    def xz(self):
        """Return xz."""
        return self.axis_matrix[0, 2]

    @property
    def volume(self) -> float:
        """Return volume."""
        return np.linalg.det(self.axis_matrix)

    def recast_types(self, uniq_elements: Union[np.ndarray, list]):
        """Recast types and related variables using unique elements."""
        if self.elements is None:
            raise AttributeError("Elements are not defined.")

        self.n_atomtypes = len(uniq_elements)
        map1 = dict()
        for i, ele in enumerate(uniq_elements):
            map1[ele] = i
        self.types = [map1[ele] for ele in self.elements]
        return self

    def to_initial_basis(self, lmp_cartesian: np.ndarray):
        """Return fractional coordinates in initial basis."""
        if self.rotation_inverse is None:
            raise ValueError("No definition of inverse rotation.")
        return self.rotation_inverse @ lmp_cartesian
