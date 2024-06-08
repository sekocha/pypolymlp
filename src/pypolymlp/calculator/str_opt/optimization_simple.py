#!/usr/bin/env python
import copy

import numpy as np
from scipy.optimize import minimize

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.calculator.properties import Properties
from pypolymlp.utils.vasp_utils import write_poscar_file


class Minimize:

    def __init__(self, cell, pot=None, params_dict=None, coeffs=None, properties=None):

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

        params_dict = self.prop.params_dict
        if isinstance(params_dict, list):
            elements = params_dict[0]["elements"]
        else:
            elements = params_dict["elements"]

        cell = update_types([cell], elements)[0]
        self.st_dict = self.set_structure(cell)

        self.__energy = None
        self.__force = None
        self.__stress = None
        self.__relax_cell = False
        self.__res = None
        self.__n_atom = len(self.st_dict["elements"])

    def set_structure(self, cell):

        self.st_dict = copy.deepcopy(cell)
        self.st_dict["axis_inv"] = np.linalg.inv(cell["axis"])
        self.st_dict["volume"] = np.linalg.det(cell["axis"])
        return self.st_dict

    """ no cell relaxation"""

    def fun_fix_cell(self, x, args=None):

        self.to_st_dict_fix_cell(x)
        self.__energy, self.__force, _ = self.prop.eval(self.st_dict)

        if self.__energy < -1e3 * self.__n_atom:
            print("Energy =", self.__energy)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )
        return self.__energy

    def jac_fix_cell(self, x, args=None):

        prod = -self.__force.T @ self.st_dict["axis"]
        derivatives = prod.reshape(-1)
        return derivatives

    def to_st_dict_fix_cell(self, x):

        self.st_dict["positions"] = x.reshape((-1, 3)).T
        return self.st_dict

    """ with cell relaxation"""

    def fun_relax_cell(self, x, args=None):

        self.to_st_dict_relax_cell(x)
        (self.__energy, self.__force, self.__stress) = self.prop.eval(self.st_dict)

        if self.__energy < -1e8:
            print("Energy =", self.__energy)
            raise ValueError(
                "Geometry optimization failed: " "Huge negative energy value."
            )
        return self.__energy

    def jac_relax_cell(self, x, args=None):

        derivatives = np.zeros(len(x))
        derivatives[:-9] = self.jac_fix_cell(x)
        sigma = [
            [self.__stress[0], self.__stress[3], self.__stress[5]],
            [self.__stress[3], self.__stress[1], self.__stress[4]],
            [self.__stress[5], self.__stress[4], self.__stress[2]],
        ]
        derivatives_s = -np.array(sigma) @ self.st_dict["axis_inv"].T
        derivatives[-9:] = derivatives_s.reshape(-1)
        return derivatives

    def to_st_dict_relax_cell(self, x):

        x_positions, x_cells = x[:-9], x[-9:]

        self.st_dict["axis"] = x_cells.reshape((3, 3))
        self.st_dict["volume"] = np.linalg.det(self.st_dict["axis"])
        self.st_dict["axis_inv"] = np.linalg.inv(self.st_dict["axis"])
        self.st_dict["positions"] = x_positions.reshape((-1, 3)).T
        return self.st_dict

    def run(self, relax_cell=False, gtol=1e-4, method="BFGS"):
        """
        Parameters
        ----------
        method: CG, BFGS, or L-BFGS-B
        """
        print("Using", method, "method")
        self.__relax_cell = relax_cell
        options = {
            "gtol": gtol,
            "disp": True,
        }

        if relax_cell:
            fun = self.fun_relax_cell
            jac = self.jac_relax_cell
            xf = self.st_dict["positions"].T.reshape(-1)
            xs = self.st_dict["axis"].reshape(-1)
            self.__x0 = np.concatenate([xf, xs], 0)
        else:
            fun = self.fun_fix_cell
            jac = self.jac_fix_cell
            self.__x0 = self.st_dict["positions"].T.reshape(-1)

        self.__res = minimize(fun, self.__x0, method=method, jac=jac, options=options)
        self.__x0 = self.__res.x
        return self

    @property
    def structure(self):
        return self.st_dict

    @property
    def energy(self):
        return self.__res.fun

    @property
    def n_iter(self):
        return self.__res.nit

    @property
    def success(self):
        if self.__res is None:
            return False
        return self.__res.success

    @property
    def residual_forces(self):
        if self.__relax_cell:
            residual_f = -self.__res.jac[:-9].reshape((-1, 3)).T
            residual_s = -self.__res.jac[-9:].reshape((3, 3))
            return residual_f, residual_s
        return -self.__res.jac.reshape((-1, 3)).T

    def print_structure(self):
        print("Axis basis vectors:")
        for a in self.st_dict["axis"].T:
            print(" -", list(a))
        print("Fractional coordinates:")
        for p, e in zip(self.st_dict["positions"].T, self.st_dict["elements"]):
            print(" -", e, list(p))

    def write_poscar(self, filename="POSCAR_eqm"):
        write_poscar_file(self.st_dict, filename=filename)


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

    unitcell = Poscar(args.poscar).get_structure()

    print("Mode: Geometry optimization")
    print("- Considering no symmetric constraints")
    if not args.cell_relax:
        print("- Fixing cell parameters")
    else:
        print("- Relaxing cell parameters")

    minobj = Minimize(unitcell, pot=args.pot)
    print("Initial structure")
    minobj.print_structure()
    minobj.run(relax_cell=args.cell_relax, gtol=1e-5)

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
