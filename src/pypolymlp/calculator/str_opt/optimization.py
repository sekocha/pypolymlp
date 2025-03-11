"""Class for geometry optimization with symmetric constraint."""

import copy
from typing import Literal, Optional, Union

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.units import EVtoGPa
from pypolymlp.utils.spglib_utils import construct_basis_cell
from pypolymlp.utils.structure_utils import refine_positions
from pypolymlp.utils.symfc_utils import construct_basis_fractional_coordinates
from pypolymlp.utils.vasp_utils import write_poscar_file


class GeometryOptimization:
    """Class for geometry optimization."""

    def __init__(
        self,
        cell: PolymlpStructure,
        relax_cell: bool = False,
        relax_volume: bool = False,
        relax_positions: bool = True,
        with_sym: bool = True,
        pressure: float = 0.0,
        pot: str = None,
        params: Optional[Union[PolymlpParams, list[PolymlpParams]]] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
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
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        params = self._prop.params
        if isinstance(params, list):
            elements = params[0].elements
        else:
            elements = params.elements
        cell = update_types([cell], elements)[0]

        if not relax_cell and not relax_volume and not relax_positions:
            raise ValueError("No degree of freedom to be optimized.")

        if relax_volume:
            relax_cell = True
        self._relax_cell = relax_cell
        self._relax_volume = relax_volume
        self._relax_positions = relax_positions
        self._with_sym = with_sym
        self._pressure = pressure
        self._verbose = verbose

        self._basis_axis, cell_update = self._set_basis_axis(cell)
        self.structure = cell_update
        self._basis_f = self._set_basis_positions(cell)

        if not self._relax_cell and not self._relax_volume:
            if not self._relax_positions:
                raise ValueError("No degree of freedom to be optimized.")

        self._positions_f0 = copy.deepcopy(self._structure.positions)
        self._x0 = self._set_initial_coefficients()
        if not relax_volume:
            self._v0 = self._structure.volume

        self._energy = None
        self._force = None
        self._stress = None
        self._res = None
        self._n_atom = len(self._structure.elements)

        if verbose:
            e0, f0, _ = self._prop.eval(self._structure)
            print("Energy (Initial structure):", e0, flush=True)

    def _set_basis_axis(self, cell: PolymlpStructure):
        """Set basis vectors for axis components."""
        if self._relax_cell:
            if self._with_sym:
                self._basis_axis, cell_update = construct_basis_cell(
                    cell,
                    verbose=self._verbose,
                )
            else:
                self._basis_axis = np.eye(9)
                cell_update = cell
        else:
            self._basis_axis = None
            cell_update = cell
        return self._basis_axis, cell_update

    def _set_basis_positions(self, cell: PolymlpStructure):
        """Set basis vectors for atomic positions."""
        if self._relax_positions:
            if self._with_sym:
                self._basis_f = construct_basis_fractional_coordinates(cell)
                if self._basis_f is None:
                    self._relax_positions = False
            else:
                N3 = cell.positions.shape[0] * cell.positions.shape[1]
                self._basis_f = np.eye(N3)
        else:
            self._basis_f = None
        return self._basis_f

    def _set_initial_coefficients(self):
        """Set initial coefficients representing structure."""
        xf, xs = [], []
        if self._relax_positions:
            xf = np.zeros(self._basis_f.shape[1])
        if self._relax_cell:
            xs = self._basis_axis.T @ self._structure.axis.reshape(-1)

        self._x0 = np.concatenate([xf, xs], 0)
        self._size_pos = 0 if self._basis_f is None else self._basis_f.shape[1]
        return self._x0

    def split(self, x: np.ndarray):
        """Split coefficients."""
        partition1 = self._size_pos
        x_pos = x[:partition1]
        x_axis = x[partition1:]
        return x_pos, x_axis

    def fun_fix_cell(self, x, args=None):
        """Target function when performing no cell optimization."""
        self._to_structure_fix_cell(x)
        self._energy, self._force, _ = self._prop.eval(self._structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy =", self._energy, flush=True)
            print("Axis :", flush=True)
            print(self._structure.axis.T, flush=True)
            print("Fractional coordinates:", flush=True)
            print(self._structure.positions.T, flush=True)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )

        self._energy += self._pressure * self._structure.volume / EVtoGPa
        return self._energy

    def jac_fix_cell(self, x, args=None):
        """Target Jacobian function when performing no cell optimization."""
        if self._basis_f is not None:
            prod = -self._force.T @ self._structure.axis
            derivatives = self._basis_f.T @ prod.reshape(-1)
            return derivatives
        return []

    def fun_relax_cell(self, x, args=None):
        """Target function when performing cell optimization."""

        self._to_structure_relax_cell(x)
        (self._energy, self._force, self._stress) = self._prop.eval(self._structure)

        if (
            self._energy < -1e3 * self._n_atom
            or abs(self._structure.volume) / self._n_atom > 1000
        ):
            print("Energy =", self._energy, flush=True)
            print("Axis :", flush=True)
            print(self._structure.axis.T, flush=True)
            print("Fractional coordinates:", flush=True)
            print(self._structure.positions.T, flush=True)
            raise ValueError(
                "Geometry optimization failed: Huge negative energy value"
                "or huge volume value."
            )

        self._energy += self._pressure * self._structure.volume / EVtoGPa
        return self._energy

    def jac_relax_cell(self, x, args=None):
        """Target Jacobian function when performing cell optimization."""
        partition1 = self._size_pos
        derivatives = np.zeros(len(x))
        if self._relax_positions:
            derivatives[:partition1] = self.jac_fix_cell(x[:partition1])
        derivatives[partition1:] = self.derivatives_by_axis()
        return derivatives

    def _to_structure_fix_cell(self, x):
        """Convert x to structure."""
        if self._basis_f is not None:
            disps_f = (self._basis_f @ x).reshape(-1, 3).T
            self._change_positions(self._positions_f0 + disps_f)
        return self._structure

    def _to_structure_relax_cell(self, x):
        """Convert x to structure."""
        x_positions, x_cells = self.split(x)
        axis = self._basis_axis @ x_cells
        axis = axis.reshape((3, 3))
        self._change_axis(axis)

        if self._relax_positions:
            self._structure = self._to_structure_fix_cell(x_positions)

        return self._structure

    def _to_volume(self, x):
        _, x_cells = self.split(x)
        axis = self._basis_axis @ x_cells
        axis = axis.reshape((3, 3))
        volume = np.linalg.det(axis)
        return volume

    def derivatives_by_axis(self):
        """Compute derivatives with respect to axis elements.

        PV @ axis_inv.T is exactly the same as the derivatives of PV term
        with respect to axis components.
        """
        pv = self._pressure * self._structure.volume / EVtoGPa
        sigma = [
            [self._stress[0] - pv, self._stress[3], self._stress[5]],
            [self._stress[3], self._stress[1] - pv, self._stress[4]],
            [self._stress[5], self._stress[4], self._stress[2] - pv],
        ]
        derivatives_s = -np.array(sigma) @ self._structure.axis_inv.T

        """derivatives_s: In the order of ax, bx, cx, ay, by, cy, az, bz, cz"""
        return self._basis_axis.T @ derivatives_s.reshape(-1)

    def run(
        self,
        method: Literal["BFGS", "CG", "L-BFGS-B", "SLSQP"] = "BFGS",
        gtol: float = 1e-4,
        maxiter: int = 1000,
        c1: Optional[float] = None,
        c2: Optional[float] = None,
    ):
        """Run geometry optimization.

        Parameters
        ----------
        method: Optimization method, CG, BFGS, L-BFGS-B or SLSQP.
                If relax_volume = False, SLSQP is automatically used.
        gtol: Tolerance for gradients.
        maxiter: Maximum iteration in scipy optimization.
        c1: c1 parameter in scipy optimization.
        c2: c2 parameter in scipy optimization.
        """
        if self._relax_cell and not self._relax_volume:
            method = "SLSQP"

        if self._verbose:
            print("Using", method, "method", flush=True)
            print("Relax cell shape:       ", self._relax_cell, flush=True)
            print("Relax volume:           ", self._relax_volume, flush=True)
            print("Relax atomic positionss:", self._relax_positions, flush=True)

        if method == "SLSQP":
            options = {"ftol": gtol, "disp": True}
        else:
            options = {"gtol": gtol, "disp": True}
            if maxiter is not None:
                options["maxiter"] = maxiter
            if c1 is not None:
                options["c1"] = c1
            if c2 is not None:
                options["c2"] = c2

        if self._relax_cell:
            fun = self.fun_relax_cell
            jac = self.jac_relax_cell
        else:
            fun = self.fun_fix_cell
            jac = self.jac_fix_cell

        if self._verbose:
            print("Number of degrees of freedom:", len(self._x0), flush=True)

        if self._relax_cell and not self._relax_volume:
            nlc = NonlinearConstraint(
                self._to_volume,
                self._v0 - 1e-15,
                self._v0 + 1e-15,
                jac="2-point",
            )
            self._res = minimize(
                fun,
                self._x0,
                method=method,
                jac=jac,
                options=options,
                constraints=[nlc],
            )
        else:
            self._res = minimize(fun, self._x0, method=method, jac=jac, options=options)
        self._x0 = self._res.x
        return self

    @property
    def relax_cell(self):
        return self._relax_cell

    @property
    def relax_volume(self):
        return self._relax_volume

    @property
    def relax_positions(self):
        return self._relax_positions

    @property
    def structure(self):
        self._structure = refine_positions(self._structure)
        return self._structure

    @structure.setter
    def structure(self, st: PolymlpStructure):
        self._structure = refine_positions(st)
        self._structure.axis_inv = np.linalg.inv(self._structure.axis)
        self._structure.volume = np.linalg.det(self._structure.axis)

    def _change_axis(self, axis: np.ndarray):
        self._structure.axis = axis
        self._structure.volume = np.linalg.det(axis)
        self._structure.axis_inv = np.linalg.inv(axis)
        return self

    def _change_positions(self, positions: np.ndarray):
        self._structure.positions = positions
        self._structure = refine_positions(self._structure)
        return self

    @property
    def energy(self):
        """Return energy at final iteration."""
        return self._res.fun

    @property
    def n_iter(self):
        """Return number of iterations."""
        return self._res.nit

    @property
    def success(self):
        """Return whether optimization is successful or not."""
        if self._res is None:
            return False
        return self._res.success

    @property
    def residual_forces(self):
        """Return residual forces and stresses represented in basis sets."""
        if self._relax_cell:
            residual_f = -self._res.jac[: self._size_pos]
            residual_s = -self._res.jac[self._size_pos :]
            return residual_f, residual_s
        return -self._res.jac

    def print_structure(self):
        """Print structure."""
        structure = self.structure
        print("Axis basis vectors:", flush=True)
        for a in structure.axis.T:
            print(" -", list(a), flush=True)
        print("Fractional coordinates:", flush=True)
        for p, e in zip(structure.positions.T, structure.elements):
            print(" -", e, list(p), flush=True)

    def write_poscar(self, filename: str = "POSCAR_eqm"):
        """Save structure to a POSCAR file."""
        write_poscar_file(self._structure, filename=filename)


if __name__ == "__main__":

    import argparse

    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", type=str, default=None, help="poscar file")
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.yaml",
        help="polymlp file",
    )
    parser.add_argument(
        "--pressure", type=float, default=0.0, help="Pressure term (in GPa)"
    )
    parser.add_argument(
        "--relax_cell", action="store_true", help="Relax cell parameters"
    )
    parser.add_argument("--relax_volume", action="store_true", help="Relax volume")
    parser.add_argument(
        "--fix_positions", action="store_true", help="Fix atomic positions"
    )
    parser.add_argument("--symmetry", action="store_true", help="Consider symmetry")
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).structure

    np.set_printoptions(legacy="1.21")

    minobj = GeometryOptimization(
        unitcell,
        relax_cell=args.relax_cell,
        relax_volume=args.relax_volume,
        relax_positions=not args.fix_positions,
        with_sym=args.symmetry,
        pressure=args.pressure,
        pot=args.pot,
        verbose=True,
    )

    print("Initial structure")
    minobj.print_structure()
    minobj.run(gtol=1e-5, method="BFGS")

    if not minobj.relax_cell:
        print("Residuals (force):")
        print(minobj.residual_forces.T)
    else:
        res_f, res_s = minobj.residual_forces
        print("Residuals (force):")
        print(res_f.T)
        print("Residuals (stress):")
        print(res_s)
