"""Class for geometry optimization with symmetric constraint."""

import copy
import sys
from typing import Literal, Optional, Union

import numpy as np
from scipy.optimize import minimize

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.symmetry import (
    basis_cell,
    construct_basis_fractional_coordinates,
)
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.utils.structure_utils import refine_positions
from pypolymlp.utils.vasp_utils import write_poscar_file


class MinimizeSym:
    """Class for geometry optimization with symmetric constraint."""

    def __init__(
        self,
        cell: PolymlpStructure,
        relax_cell: bool = False,
        relax_positions: bool = True,
        pot: str = None,
        params: Optional[Union[PolymlpParams, list[PolymlpParams]]] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        cell: Initial structure.
        relax_cell: Optimize cell shape.
        relax_positions: Optimize atomic positions.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        params = self.prop.params
        if isinstance(params, list):
            elements = params[0].elements
        else:
            elements = params.elements

        self._verbose = verbose
        cell = update_types([cell], elements)[0]

        self._relax_cell = relax_cell
        self._relax_positions = relax_positions

        if relax_cell:
            self._basis_axis, cell_update = basis_cell(cell)
            if not np.allclose(cell.axis, cell_update.axis):
                if self._verbose:
                    print("- Input structure is standarized by spglib.")
        else:
            self._basis_axis = None
            cell_update = cell

        self.structure = cell_update

        self._basis_f = None
        self._split = 0
        if relax_positions:
            self._basis_f = construct_basis_fractional_coordinates(self.structure)
            if self._basis_f is None:
                self._relax_positions = False
            else:
                self._split = self._basis_f.shape[1]

        if self._relax_cell is False and self._relax_positions is False:
            raise ValueError("No degree of freedom to be optimized.")

        self._positions_f0 = copy.deepcopy(self.structure.positions)
        self.structure = self._set_structure(self.structure)

        self._energy = None
        self._force = None
        self._stress = None
        self._res = None
        self._n_atom = len(self.structure.elements)

    def _set_structure(self, cell: PolymlpStructure):

        self.structure = copy.deepcopy(cell)
        self.structure.axis_inv = np.linalg.inv(cell.axis)
        self.structure.volume = np.linalg.det(cell.axis)
        self._set_x0()
        return self.structure

    def _set_x0(self):

        if self._relax_cell:
            xs = self._basis_axis.T @ self.structure.axis.reshape(-1)
            if self._relax_positions:
                xf = np.zeros(self._basis_f.shape[1])
                self._x0 = np.concatenate([xf, xs], 0)
            else:
                self._x0 = xs
        else:
            if self._relax_positions:
                self._x0 = np.zeros(self._basis_f.shape[1])
            else:
                raise ValueError("No degree of freedom to be optimized.")
        return self._x0

    def fun_fix_cell(self, x, args=None):
        """Target function when performing no cell optimization."""
        self.to_st_dict_fix_cell(x)
        self._energy, self._force, _ = self.prop.eval(self.structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy =", self._energy)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )

        return self._energy

    def jac_fix_cell(self, x, args=None):
        """Target Jacobian function when performing no cell optimization."""
        if self._basis_f is not None:
            prod = -self._force.T @ self.structure.axis
            derivatives = self._basis_f.T @ prod.reshape(-1)
            return derivatives
        return []

    def to_st_dict_fix_cell(self, x):
        """Convert x to structure."""
        if self._basis_f is not None:
            disps_f = (self._basis_f @ x).reshape(-1, 3).T
            self.structure.positions = self._positions_f0 + disps_f
        return self.structure

    def fun_relax_cell(self, x, args=None):
        """Target function when performing cell optimization."""

        self.to_st_dict_relax_cell(x)
        (self._energy, self._force, self._stress) = self.prop.eval(self.structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy =", self._energy)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )
        return self._energy

    def jac_relax_cell(self, x, args=None):
        """Target Jacobian function when performing cell optimization."""

        derivatives = np.zeros(len(x))
        if self._relax_positions:
            derivatives[: self._split] = self.jac_fix_cell(x)
        derivatives[self._split :] = self.derivatives_by_axis()
        return derivatives

    def to_st_dict_relax_cell(self, x):
        """Convert x to structure."""
        x_positions, x_cells = x[: self._split], x[self._split :]

        axis = self._basis_axis @ x_cells
        axis = axis.reshape((3, 3))
        self.structure.axis = axis
        self.structure.volume = np.linalg.det(axis)
        self.structure.axis_inv = np.linalg.inv(axis)

        if self._relax_positions:
            self.structure = self.to_st_dict_fix_cell(x_positions)
        return self.structure

    def derivatives_by_axis(self):
        """Compute derivatives with respect to axis elements."""
        sigma = [
            [self._stress[0], self._stress[3], self._stress[5]],
            [self._stress[3], self._stress[1], self._stress[4]],
            [self._stress[5], self._stress[4], self._stress[2]],
        ]
        derivatives_s = -np.array(sigma) @ self.structure.axis_inv.T

        """derivatives_s: In the order of ax, bx, cx, ay, by, cy, az, bz, cz"""
        return self._basis_axis.T @ derivatives_s.reshape(-1)

    def run(
        self,
        gtol: float = 1e-4,
        method: Literal["BFGS", "CG", "L-BFGS-B"] = "BFGS",
    ):
        """Run geometry optimization.

        Parameters
        ----------
        method: Optimization method, CG, BFGS, or L-BFGS-B.
        """
        if self._verbose:
            print("Using", method, "method")
        options = {"gtol": gtol, "disp": True}
        if self._relax_cell:
            fun = self.fun_relax_cell
            jac = self.jac_relax_cell
        else:
            fun = self.fun_fix_cell
            jac = self.jac_fix_cell

        if self._verbose:
            print("Number of degrees of freedom:", len(self._x0))
        self._res = minimize(fun, self._x0, method=method, jac=jac, options=options)
        self._x0 = self._res.x
        return self

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, st: PolymlpStructure):
        self._structure = refine_positions(st)

    @property
    def energy(self):
        return self._res.fun

    @property
    def n_iter(self):
        return self._res.nit

    @property
    def success(self):
        if self._res is None:
            return False
        return self._res.success

    @property
    def residual_forces(self):
        if self._relax_cell:
            residual_f = -self._res.jac[: self._split]
            residual_s = -self._res.jac[self._split :]
            return residual_f, residual_s
        return -self._res.jac

    def print_structure(self):
        self.structure = refine_positions(self.structure)
        print("Axis basis vectors:")
        for a in self.structure.axis.T:
            print(" -", list(a))
        print("Fractional coordinates:")
        for p, e in zip(self.structure.positions.T, self.structure.elements):
            print(" -", e, list(p))

    def write_poscar(self, filename="POSCAR_eqm"):
        write_poscar_file(self.structure, filename=filename)


if __name__ == "__main__":

    import argparse

    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", type=str, default=None, help="poscar file")
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.lammps",
        help="polymlp file",
    )
    parser.add_argument(
        "--cell_relax", action="store_true", help="Relaxing cell parameters"
    )
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).structure

    print("Mode: Geometry optimization")
    print("- Considering symmetric constraints")
    if not args.cell_relax:
        print("- Fixing cell parameters")
        try:
            minobj = MinimizeSym(unitcell, pot=args.pot)
        except ValueError:
            print("No degree of freedom to be optimized.")
            sys.exit(8)
    else:
        print("- Relaxing cell parameters")
        minobj = MinimizeSym(unitcell, pot=args.pot, relax_cell=True)

    print("Initial structure")
    minobj.print_structure()
    minobj.run(gtol=1e-5)

    if not args.cell_relax:
        print("Residuals (force):")
        print(minobj.residual_forces.T)
    else:
        res_f, res_s = minobj.residual_forces
        print("Residuals (force):")
        print(res_f.T)
        print("Residuals (stress):")
        print(res_s)

    print("Final structure")
    minobj.print_structure()
    minobj.write_poscar()
