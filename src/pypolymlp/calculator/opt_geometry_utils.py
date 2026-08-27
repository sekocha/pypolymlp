"""Class for basis set used in geometry optimization."""

import copy
from typing import Optional

import numpy as np
from symfc.api_symfc import eigh

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.structure_utils import _refine_positions
from pypolymlp.utils.symfc_utils import construct_basis_fractional_coordinates


class BasisSetGO:
    """Class for basis set used in geometry optimization."""

    def __init__(
        self,
        cell: PolymlpStructure,
        elements: tuple | list,
        relax_cell: bool = False,
        relax_volume: bool = False,
        relax_positions: bool = True,
        with_sym: bool = True,
        selective_dynamics_cell: Optional[np.ndarray] = None,
        selective_dynamics_positions: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        cell: Initial structure.
        relax_cell: Optimize cell shape.
        relax_volume: Optimize volume.
        relax_positions: Optimize atomic positions.
        with_sym: Consider symmetric properties.
        pressure: Pressure in GPa.
        selective_dynamics_cell: Selective dynamics for cell.
                                (3, 3) array with bool elements.
        selective_dynamics_positions: Selective dynamics for positions.
                                (3, N) array with bool elements.
        """
        self._relax_cell = relax_cell
        self._relax_volume = relax_volume
        self._relax_positions = relax_positions
        self._with_sym = with_sym
        self._verbose = verbose

        self._basis_a = None
        self._basis_f = None

        self._init_structure = update_types(cell, elements)
        self._basis_a, self._init_structure = self._set_basis_axis(self._init_structure)
        self._basis_f = self._set_basis_positions(self._init_structure)

        self._basis_a = self._apply_sd_cell(selective_dynamics_cell)
        self._basis_f = self._apply_sd_positions(selective_dynamics_positions)

        if self._basis_a is None and self._basis_f is None:
            raise RuntimeError("No degree of freedom to be optimized.")

        self._a0 = self._init_structure.axis
        self._f0 = self._init_structure.positions
        self._v0 = np.linalg.det(self._init_structure.axis)

        self._basis_size_f = 0 if self._basis_f is None else self._basis_f.shape[1]
        self._basis_size = 0
        if self._basis_f is not None:
            self._basis_size += self._basis_f.shape[1]
        if self._basis_a is not None:
            self._basis_size += self._basis_a.shape[1]

        if self._verbose:
            self._print_basis()

    def _print_basis(self):
        """Print basis sets."""
        print("Relax cell shape:       ", self._relax_cell, flush=True)
        print("Relax volume:           ", self._relax_volume, flush=True)
        print("Relax atomic positions: ", self._relax_positions, flush=True)
        print("Basis (Axis)", flush=True)
        print(self._basis_a, flush=True)
        print("Basis (Positions)", flush=True)
        print(self._basis_f, flush=True)
        print("Degrees of freedom:", self._basis_size, flush=True)

    def _set_basis_axis(self, cell: PolymlpStructure):
        """Set basis vectors for axis components."""
        if not self._relax_cell and not self._relax_volume:
            return None, cell
        if not self._relax_cell and self._relax_volume:
            return np.full((9, 1), 1.0 / 3.0), cell
        if not self._with_sym:
            return np.eye(9), cell

        basis_a, cell_update = construct_basis_cell(cell, verbose=self._verbose)
        return basis_a, cell_update

    def _set_basis_positions(self, cell: PolymlpStructure):
        """Set basis vectors for atomic positions."""
        if not self._relax_positions:
            return None
        if not self._with_sym:
            return np.eye(cell.positions.size)

        basis_f = construct_basis_fractional_coordinates(cell)
        return basis_f

    def _apply_sd_cell(self, sd_cell: np.ndarray | None):
        """Apply selective dynamics."""
        if sd_cell is None or self._basis_a is None:
            return self._basis_a

        if sd_cell.shape != (3, 3):
            raise RuntimeError("Shape of selective_dynamics_cell != (3, 3).")

        sd = sd_cell.reshape(-1)
        proj_a = self._basis_a @ self._basis_a.T
        proj_a[~sd, :] = 0
        proj_a[:, ~sd] = 0
        basis_a = eigh(proj_a)
        return basis_a

    def _apply_sd_positions(self, sd_positions: np.ndarray | None):
        """Apply selective dynamics."""
        if sd_positions is None or self._basis_f is None:
            return self._basis_f

        n_atom = len(self.structure.elements)
        if sd_positions.shape != (3, n_atom):
            raise RuntimeError("Shape of selective_dynamics_cell != (3, n_atom).")

        sd = sd_positions.T.reshape(-1)
        proj_f = self._basis_f @ self._basis_f.T
        proj_f[~sd, :] = 0
        proj_f[:, ~sd] = 0
        basis_f = eigh(proj_f)
        return basis_f

    @property
    def basis_a(self):
        """Return basis set for axis."""
        return self._basis_a

    @property
    def basis_f(self):
        """Return basis set for fractional coordinates.

        The row order is (1, xa), (1, xb), ..., (N, xb), (N, xc).
        """
        return self._basis_f

    @property
    def init_structure(self):
        """Return initial structure."""
        return self._init_structure

    @property
    def init_coeffs(self):
        """Set initial coefficients representing structure."""
        return np.zeros(self._basis_size)

    def axis(self, x: np.ndarray):
        """Convert coeffs. to axis."""
        if self._basis_a is None:
            return self._a0

        da = self._basis_a @ x
        da = da.reshape((3, 3))
        return self._a0 + da

    def positions(self, x: np.ndarray):
        """Convert coeffs. to fractional coordinates."""
        if self._basis_f is None:
            return self._f0
        df = (self._basis_f @ x).reshape(-1, 3).T
        return _refine_positions(self._f0 + df)

    def split(self, x: np.ndarray):
        """Split coefficients."""
        partition1 = self._basis_size_f
        x_pos = x[:partition1]
        x_axis = x[partition1:]
        return x_pos, x_axis

    def structure(self, x: np.ndarray):
        """Convert coeffs. to structure."""
        x_pos, x_cell = self.split(x)
        axis = self.axis(x_cell)
        positions = self.positions(x_pos)
        st = copy.deepcopy(self._init_structure)
        st.axis = axis
        st.positions = positions
        return st


#    def _to_volume(self, x: np.ndarray):
#        """Calculate volume from variable vector."""
#        _, x_cells = self.split(x)
#        axis = self._basis_axis @ x_cells
#        axis = axis.reshape((3, 3)) + self._axis0
#        volume = np.linalg.det(axis)
#        return volume
#
#    def fun_fix_cell(self, x, args=None):
#        """Target function when performing no cell optimization."""
#        self._to_structure_fix_cell(x)
#        self._energy, self._force, _ = self._prop.eval(self._structure)
#
#        if self._energy < -1e3 * self._n_atom:
#            print("Energy =", self._energy, flush=True)
#            print("Axis :", flush=True)
#            print(self._structure.axis.T, flush=True)
#            print("Fractional coordinates:", flush=True)
#            print(self._structure.positions.T, flush=True)
#            raise ValueError(
#                "Geometry optimization failed: " "Huge negative energy value."
#            )
#
#        self._energy += self._pressure * self._structure.volume / EVtoGPa
#        return self._energy
#
#    def jac_fix_cell(self, x, args=None):
#        """Target Jacobian function when performing no cell optimization."""
#        if self._basis_f is not None:
#            prod = -self._force.T @ self._structure.axis
#            derivatives = self._basis_f.T @ prod.reshape(-1)
#            return derivatives
#        return []
#
#    def fun_relax_cell(self, x, args=None):
#        """Target function when performing cell optimization."""
#
#        self._to_structure_relax_cell(x)
#        (self._energy, self._force, self._stress) = self._prop.eval(self._structure)
#
#        if (
#            self._energy < -1e3 * self._n_atom
#            or abs(self._structure.volume) / self._n_atom > 1000
#        ):
#            print("Energy =", self._energy, flush=True)
#            print("Axis :", flush=True)
#            print(self._structure.axis.T, flush=True)
#            print("Fractional coordinates:", flush=True)
#            print(self._structure.positions.T, flush=True)
#            raise ValueError(
#                "Geometry optimization failed: Huge negative energy value"
#                "or huge volume value."
#            )
#
#        self._energy += self._pressure * self._structure.volume / EVtoGPa
#        return self._energy
#
#    def jac_relax_cell(self, x, args=None):
#        """Target Jacobian function when performing cell optimization."""
#        partition1 = self._size_pos
#        derivatives = np.zeros(len(x))
#        if self._relax_positions:
#            derivatives[:partition1] = self.jac_fix_cell(x[:partition1])
#        derivatives[partition1:] = self.derivatives_by_axis()
#        return derivatives
#
#    def derivatives_by_axis(self):
#        """Compute derivatives with respect to axis elements.
#
#        PV @ axis_inv.T is exactly the same as the derivatives of PV term
#        with respect to axis components.
#
#        Under the constraint of a fixed cell shape, the mean normal stress
#        serves as an approximation to the derivative of the enthalpy
#        with respect to volume.
#        """
#        pv = self._pressure * self._structure.volume / EVtoGPa
#        sigma = [
#            [self._stress[0] - pv, self._stress[3], self._stress[5]],
#            [self._stress[3], self._stress[1] - pv, self._stress[4]],
#            [self._stress[5], self._stress[4], self._stress[2] - pv],
#        ]
#        if self._relax_cell:
#            """derivatives_s: In the order of ax, bx, cx, ay, by, cy, az, bz, cz"""
#            derivatives_s = -np.array(sigma) @ self._structure.axis_inv.T
#            derivatives_s = self._basis_axis.T @ derivatives_s.reshape(-1)
#        else:
#            derivatives_s = -np.trace(np.array(sigma)) / 3
#
#        return derivatives_s
#
