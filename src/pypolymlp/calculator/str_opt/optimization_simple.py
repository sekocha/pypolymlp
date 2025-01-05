"""Class for geometry optimization without symmetric constraint."""

import copy
from typing import Literal, Optional, Union

import numpy as np
from scipy.optimize import minimize

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.utils.structure_utils import refine_positions
from pypolymlp.utils.vasp_utils import write_poscar_file


class Minimize:

    def __init__(
        self,
        cell: PolymlpStructure,
        pot: str = None,
        params: Optional[Union[PolymlpParams, list[PolymlpParams]]] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        relax_cell: bool = True,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        cell: Initial structure.
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

        self._structure = self._set_structure(cell)

        self._energy = None
        self._force = None
        self._stress = None
        self._relax_cell = relax_cell
        self._res = None
        self._n_atom = len(self._structure.elements)

        if verbose:
            e0, f0, _ = self.prop.eval(self._structure)
            print("Energy (Initial structure):", e0)

    def _set_structure(self, cell: PolymlpStructure):

        self._structure = copy.deepcopy(cell)
        self._structure = refine_positions(self._structure)
        self._structure.axis_inv = np.linalg.inv(cell.axis)
        self._structure.volume = np.linalg.det(cell.axis)
        return self._structure

    def fun_fix_cell(self, x, args=None):
        """Target function when performing no cell optimization."""
        self._to_structure_fix_cell(x)
        self._energy, self._force, _ = self.prop.eval(self._structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy :", self._energy)
            print("Axis :")
            print(self._structure.axis.T)
            print("Fractional coordinates:")
            print(self._structure.positions.T)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )
        return self._energy

    def jac_fix_cell(self, x, args=None):
        """Target Jacobian function when performing no cell optimization."""
        prod = -self._force.T @ self._structure.axis
        derivatives = prod.reshape(-1)
        return derivatives

    def _to_structure_fix_cell(self, x):
        """Convert x to structure."""
        self._structure.positions = x.reshape((-1, 3)).T
        self._structure = refine_positions(self._structure)
        return self._structure

    def fun_relax_cell(self, x, args=None):
        """Target function when performing cell optimization."""

        self._to_structure_relax_cell(x)
        (self._energy, self._force, self._stress) = self.prop.eval(self._structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy =", self._energy)
            print("Axis :")
            print(self._structure.axis.T)
            print("Fractional coordinates:")
            print(self._structure.positions.T)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )
        return self._energy

    def jac_relax_cell(self, x, args=None):
        """Target Jacobian function when performing cell optimization."""

        derivatives = np.zeros(len(x))
        derivatives[:-9] = self.jac_fix_cell(x)
        sigma = [
            [self._stress[0], self._stress[3], self._stress[5]],
            [self._stress[3], self._stress[1], self._stress[4]],
            [self._stress[5], self._stress[4], self._stress[2]],
        ]
        derivatives_s = -np.array(sigma) @ self._structure.axis_inv.T
        derivatives[-9:] = derivatives_s.reshape(-1)
        return derivatives

    def _to_structure_relax_cell(self, x):
        """Convert x to structure."""
        x_positions, x_cells = x[:-9], x[-9:]

        self._structure.axis = x_cells.reshape((3, 3))
        self._structure.volume = np.linalg.det(self._structure.axis)
        self._structure.axis_inv = np.linalg.inv(self._structure.axis)
        self._structure.positions = x_positions.reshape((-1, 3)).T
        self._structure = refine_positions(self._structure)
        return self._structure

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
            xf = self._structure.positions.T.reshape(-1)
            xs = self._structure.axis.reshape(-1)
            self._x0 = np.concatenate([xf, xs], 0)
            self._x_prev = copy.deepcopy(self._x0)
        else:
            fun = self.fun_fix_cell
            jac = self.jac_fix_cell
            self._x0 = self._structure.positions.T.reshape(-1)
            self._x_prev = copy.deepcopy(self._x0)

        self._res = minimize(fun, self._x0, method=method, jac=jac, options=options)
        self._x0 = self._res.x
        return self

    @property
    def structure(self):
        self._structure = refine_positions(self._structure)
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
            residual_f = -self._res.jac[:-9].reshape((-1, 3)).T
            residual_s = -self._res.jac[-9:].reshape((3, 3))
            return residual_f, residual_s
        return -self._res.jac.reshape((-1, 3)).T

    def print_structure(self):
        structure = self.structure
        print("Axis basis vectors:")
        for a in structure.axis.T:
            print(" -", list(a))
        print("Fractional coordinates:")
        for p, e in zip(structure.positions.T, structure.elements):
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

    np.set_printoptions(legacy="1.21")
    print("Mode: Geometry optimization")
    print("- Considering no symmetric constraints")
    if not args.cell_relax:
        print("- Fixing cell parameters")
    else:
        print("- Relaxing cell parameters")

    minobj = Minimize(unitcell, pot=args.pot, relax_cell=args.cell_relax)
    print("Initial structure")
    minobj.print_structure()
    minobj.run(gtol=1e-5, method="BFGS")

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
