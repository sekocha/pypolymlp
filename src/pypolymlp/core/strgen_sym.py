"""Class for generating structures satisfying symmetric properties."""

import os

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.structure_utils import refine_positions
from pypolymlp.utils.symfc_utils import construct_basis_cartesian
from pypolymlp.utils.vasp_utils import write_poscar_file


class StructureGeneratorSym:
    """Class for generating structures satisfying symmetric properties."""

    def __init__(
        self,
        cell: PolymlpStructure,
        fix_axis: bool = False,
        fix_positions: bool = False,
        verbose: bool = False,
    ):
        """Init method."""
        if fix_axis and fix_positions:
            raise RuntimeError("Both axis and positions fixed.")

        self._cell = cell
        self._fix_axis = fix_axis
        self._fix_positions = fix_positions
        self._verbose = verbose

        if not fix_axis:
            self._basis_axis, self._cell = construct_basis_cell(
                self._cell,
                verbose=verbose,
            )
            self._init_axis_coeffs = self._basis_axis.T @ self._cell.axis.reshape(-1)

        if not fix_positions:
            self._basis_cartesian = construct_basis_cartesian(self._cell)
            if fix_axis and self._basis_cartesian is None:
                raise RuntimeError("No degree of freedom found.")

        self._natom = self._cell.positions.shape[1]
        self._structures = []

    def _recover_axis(self, coeffs):
        return (self._basis_axis @ coeffs).reshape((3, 3, -1)).transpose((2, 0, 1))

    def _recover_displacements(self, coeffs):
        disps = self._basis_cartesian @ coeffs
        return disps.reshape((self._natom, 3, -1)).transpose((2, 1, 0))

    def _random_deformation(self, n_samples: int = 100, max_deform: float = 0.1):
        """Deform axis randomly."""
        coeffs_deform = np.random.rand(self._basis_axis.shape[1], n_samples)
        coeffs_deform = (coeffs_deform - 0.5) * 2 * max_deform
        return self._recover_axis(coeffs_deform + self._init_axis_coeffs[:, None])

    def _random_displacements(self, n_samples: int = 100, max_distance: float = 0.1):
        """Apply random atomic displacements."""
        coeffs_disps = np.random.rand(self._basis_cartesian.shape[1], n_samples)
        coeffs_disps = (coeffs_disps - 0.5) * 2 * max_distance
        return self._recover_displacements(coeffs_disps)

    def run(
        self,
        n_samples: int = 100,
        max_deform: float = 0.1,
        max_distance: float = 0.1,
    ):
        """Generate random structures with symmetry constraints."""
        if not self._fix_axis:
            axis_all = self._random_deformation(
                n_samples=n_samples,
                max_deform=max_deform,
            )
        else:
            axis_all = [self._cell.axis for i in range(n_samples)]

        if not self._fix_positions and self._basis_cartesian is not None:
            disps_all = self._random_displacements(
                n_samples=n_samples,
                max_distance=max_distance,
            )
        else:
            shape = self._cell.positions.shape
            disps_all = [np.zeros(shape) for i in range(n_samples)]

        for axis, disps in zip(axis_all, disps_all):
            disps_f = np.linalg.inv(axis) @ disps
            positions = self._cell.positions + disps_f
            st = PolymlpStructure(
                axis=axis,
                positions=positions,
                n_atoms=self._cell.n_atoms,
                types=self._cell.types,
                elements=self._cell.elements,
            )
            self._structures.append(refine_positions(st))
        return self

    def save_structures(self, path="poscars"):
        """Save structures in POSCAR format."""
        os.makedirs(path, exist_ok=True)
        for i, st in enumerate(self._structures):
            write_poscar_file(st, filename=path + "/POSCAR-" + str(i + 1).zfill(4))
        return self

    @property
    def structures(self):
        """Return generated structures."""
        return self._structures

    @property
    def basis_sets(self):
        """Return basis sets for axis and positions.

        Returns
        -------
        basis_axis: Basis set for representing axis matrix, shape=(n_basis, 3, 3).
        basis_cartesian: Basis set for atomic displacements in Cartesian,
                         shape=(n_basis, 3, n_atom).
        """
        if self._basis_cartesian is None:
            basis_reshape = None
        else:
            basis_reshape = self._basis_cartesian.reshape(
                (self._natom, 3, -1)
            ).transpose((2, 1, 0))
        return (
            self._basis_axis.reshape((3, 3, -1)).transpose((2, 0, 1)),
            basis_reshape,
        )
