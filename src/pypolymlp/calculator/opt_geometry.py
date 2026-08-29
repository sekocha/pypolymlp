"""Class for geometry optimization with symmetric constraint."""

from typing import Literal, Optional

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.opt_geometry_utils import BasisSetGO
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoGPa
from pypolymlp.utils.vasp_utils import write_poscar_file


class GeometryOptimization:
    """Class for geometry optimization."""

    def __init__(
        self,
        cell: PolymlpStructure,
        properties: Properties,
        relax_cell: bool = False,
        relax_volume: bool = False,
        relax_positions: bool = True,
        with_sym: bool = True,
        selective_dynamics_cell: Optional[np.ndarray] = None,
        selective_dynamics_positions: Optional[np.ndarray] = None,
        pressure: float = 0.0,
        scale_axis: Optional[float] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        cell: Initial structure.
        properties: Properties instance.
        relax_cell: Optimize cell shape.
        relax_volume: Optimize volume.
        relax_positions: Optimize atomic positions.
        with_sym: Consider symmetric properties.
        selective_dynamics_cell: Selective dynamics for cell.
                                (3, 3) array with bool elements.
        selective_dynamics_positions: Selective dynamics for positions.
                                (3, N) array with bool elements.
        pressure: Pressure in GPa.
        """
        self._prop = properties
        self._verbose = verbose
        self._structure = None
        self._pressure = pressure

        self._basis = BasisSetGO(
            cell=cell,
            elements=self._prop.elements,
            relax_cell=relax_cell,
            relax_volume=relax_volume,
            relax_positions=relax_positions,
            with_sym=with_sym,
            selective_dynamics_cell=selective_dynamics_cell,
            selective_dynamics_positions=selective_dynamics_positions,
            verbose=verbose,
        )
        self._basis_f = self._basis.basis_f
        self._basis_a = self._basis.basis_a
        self.structure = self._basis.init_structure

        self._n_atom = len(self._structure.elements)
        self._x0 = self._basis.init_coeffs
        self._basis_size = self._basis.basis_size
        self._basis_size_f = self._basis.basis_size_f
        self._v0 = self._basis._v0

        self._scale = self._set_scale(scale_axis=scale_axis)

        self._energy = None
        self._force = None
        self._stress = None
        self._res = None

        if verbose:
            e0, _, _ = self._prop.eval(self._structure)
            h0 = e0 + self._pressure * self._structure.volume / EVtoGPa
            print("---------------------------", flush=True)
            print("Initial structure", flush=True)
            self.print_structure()
            print("Energy (Initial structure):", e0, flush=True)
            print("E + PV (Initial structure):", h0, flush=True)
            print("---------------------------", flush=True)

    def _set_scale(self, scale_axis: Optional[float] = None):
        """Set scale for fractional coordinates and axis matrix elements."""
        if self._basis_a is None:
            return None

        scale = np.ones(self._basis_size)
        if scale_axis is None:
            val = pow(self._v0, 1 / 3) * 0.1
            scale[self._basis_size_f :] = val
        else:
            scale[self._basis_size_f :] = scale_axis
        return scale

    def _fun_fix_cell(self, x: np.ndarray, args=None):
        """Target function when performing no cell optimization."""
        self.structure = self._basis.structure(x)
        self._energy, self._force, _ = self._prop.eval(self._structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy =", self._energy, flush=True)
            print("Axis :", flush=True)
            print(self._structure.axis.T, flush=True)
            raise RuntimeError("Failed: Huge negative energy value.")

        self._energy += self._pressure * self._structure.volume / EVtoGPa
        return self._energy

    def _fun_relax_cell(self, x: np.ndarray, args=None):
        """Target function when performing cell optimization."""
        x_scaled = x * self._scale
        self.structure = self._basis.structure(x_scaled)
        self._energy, self._force, self._stress = self._prop.eval(self._structure)

        volume = self._structure.volume
        if self._energy < -1e3 * self._n_atom or abs(volume) / self._n_atom > 1000:
            print("Energy =", self._energy, flush=True)
            print("Axis :", flush=True)
            print(self._structure.axis.T, flush=True)
            raise RuntimeError("Failed: Huge negative energy value or huge volume.")

        self._energy += self._pressure * volume / EVtoGPa
        return self._energy

    def _fun_relax_cell_fix_volume(self, x: np.ndarray, args=None):
        """Target function when performing cell optimization."""
        x_scaled = x * self._scale
        self.structure = self._basis.structure(x_scaled)
        self._energy, self._force, self._stress = self._prop.eval(self._structure)

        if self._energy < -1e3 * self._n_atom:
            print("Energy =", self._energy, flush=True)
            print("Axis :", flush=True)
            print(self._structure.axis.T, flush=True)
            raise RuntimeError("Failed: Huge negative energy value or huge volume.")

        return self._energy

    def _jac_fix_cell(self, args=None):
        """Target Jacobian function when performing no cell optimization."""
        derivatives = self._derivatives_by_frac()
        return derivatives

    def _jac_relax_cell(self, args=None):
        """Target Jacobian function when performing cell optimization."""
        derivatives = np.zeros(self._basis_size)
        partition1 = self._basis_size_f
        if self._basis_f is not None:
            derivatives[:partition1] = self._derivatives_by_frac()
        derivatives[partition1:] = self._derivatives_by_axis()

        derivatives = derivatives * self._scale
        return derivatives

    def _jac_relax_cell_fix_volume(self, args=None):
        """Target Jacobian function when performing cell optimization."""
        derivatives = np.zeros(self._basis_size)
        partition1 = self._basis_size_f
        if self._basis_f is not None:
            derivatives[:partition1] = self._derivatives_by_frac()

        sigma = [
            [self._stress[0], self._stress[3], self._stress[5]],
            [self._stress[3], self._stress[1], self._stress[4]],
            [self._stress[5], self._stress[4], self._stress[2]],
        ]
        derivatives_s = -np.array(sigma) @ self._structure.axis_inv.T
        derivatives_s = self._basis_a.T @ derivatives_s.reshape(-1)
        derivatives[partition1:] = derivatives_s

        derivatives = derivatives * self._scale
        return derivatives

    def _derivatives_by_frac(self):
        """Compute derivatives with respect to fractional coordinates."""
        prod = -self._force.T @ self._structure.axis
        derivatives_f = self._basis_f.T @ prod.reshape(-1)
        return derivatives_f

    def _derivatives_by_axis(self):
        """Compute derivatives with respect to axis elements.

        PV @ axis_inv.T is exactly the same as the derivatives of PV term
        with respect to axis components.

        Under the constraint of a fixed cell shape, the mean normal stress
        serves as an approximation to the derivative of the enthalpy
        with respect to volume.

        The order of derivatives_s: ax, bx, cx, ay, by, cy, az, bz, cz.
        """
        pv = self._pressure * self._structure.volume / EVtoGPa
        sigma = [
            [self._stress[0] - pv, self._stress[3], self._stress[5]],
            [self._stress[3], self._stress[1] - pv, self._stress[4]],
            [self._stress[5], self._stress[4], self._stress[2] - pv],
        ]
        derivatives_s = -np.array(sigma) @ self._structure.axis_inv.T
        derivatives_s = self._basis_a.T @ derivatives_s.reshape(-1)
        return derivatives_s

    def _fun_volume(self, x: np.ndarray):
        """Function to return volume."""
        x_scaled = x * self._scale
        self.structure = self._basis.structure(x_scaled)
        return self._structure.volume

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
        use_constraint = False
        if self._basis._relax_cell and not self._basis._relax_volume:
            use_constraint = True
            method = "SLSQP"

        if self._verbose:
            print("Using", method, "method", flush=True)

        if method == "SLSQP":
            options = {"ftol": gtol * 1e-3, "eps": 1e-3, "disp": True}
        else:
            options = {"gtol": gtol, "disp": True}
            if maxiter is not None:
                options["maxiter"] = maxiter
            if c1 is not None:
                options["c1"] = c1
            if c2 is not None:
                options["c2"] = c2
        options["disp"] = self._verbose

        if self._basis_a is None:
            fun = self._fun_fix_cell
            jac = self._jac_fix_cell
        else:
            if self._basis._relax_volume:
                fun = self._fun_relax_cell
                jac = self._jac_relax_cell
            else:
                fun = self._fun_relax_cell_fix_volume
                jac = self._jac_relax_cell_fix_volume

        if use_constraint:
            nlc = NonlinearConstraint(
                self._fun_volume,
                self._v0,
                self._v0,
                jac="3-point",
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
    def structure(self):
        """Return structure."""
        return self._structure

    @structure.setter
    def structure(self, st: PolymlpStructure):
        """Setter of structure."""
        self._structure = st
        self._structure.axis_inv = np.linalg.inv(self._structure.axis)
        self._structure.volume = np.linalg.det(self._structure.axis)

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
        if self._basis_a is None:
            return -self._res.jac

        partition1 = self._basis_size_f
        residual_f = -self._res.jac[:partition1]
        residual_s = -self._res.jac[partition1:]
        return residual_f, residual_s

    def print_residuals(self):
        """Print force and stress residuals."""
        print("Residuals (force, eV/ang):", flush=True)
        print(self._force.T)
        if self._basis_a is None:
            print("Gradients (force):", flush=True)
            print(self.residual_forces.T, flush=True)
            return self

        print("Residuals (stress, eV/cell):", flush=True)
        print(self._stress)
        res_f, res_s = self.residual_forces
        print("Gradients (force):", flush=True)
        print(res_f.T, flush=True)
        print("Gradients (stress):", flush=True)
        print(res_s, flush=True)
        return self

    def print_structure(self):
        """Print structure."""
        structure = self.structure
        np.set_printoptions(suppress=True)
        print("Axis basis vectors:", flush=True)
        for a in structure.axis.T:
            print("-", list(a), flush=True)
        print("Fractional coordinates:", flush=True)
        for p, e in zip(structure.positions.T, structure.elements):
            print("-", e, list(p), flush=True)
        return self

    def write_poscar(self, filename: str = "POSCAR_eqm"):
        """Save structure to a POSCAR file."""
        write_poscar_file(self._structure, filename=filename)

    def change_basis_axis(self, basis_a: np.ndarray):
        """Change basis set for axis."""
        self._basis.basis_a = basis_a
        self._basis_a = self._basis.basis_a
        self._basis_size = self._basis.basis_size
        self._x0 = self._basis.init_coeffs
        return self
