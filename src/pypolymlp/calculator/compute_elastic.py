#!/usr/bin/env python
import copy

import numpy as np
import pymatgen as pmg
from pymatgen.analysis.elasticity import DeformedStructureSet, diff_fit

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym


class PolymlpElastic:

    def __init__(
        self,
        unitcell_dict,
        unitcell_poscar,
        pot=None,
        params_dict=None,
        coeffs=None,
        properties=None,
        geometry_optimization=True,
    ):
        """
        Parameters
        ----------
        unitcell_dict: unitcell in dict format
        pot or (params_dict and coeffs): polynomal MLP
        """

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

        self.__unitcell_dict = unitcell_dict
        fposcar = open(unitcell_poscar)
        self.st_pmg = pmg.core.Structure.from_str(fposcar.read(), fmt="POSCAR")
        fposcar.close()
        # self.__run_initial_geometry_optimization()

        self.__compute_initial_properties()

    def __compute_initial_properties(self):
        """stress: xx, yy, zz, xy, yz, zx
        --> p0: xx(1), yy(2), zz(3), yz(4), zx(5), xy(6)
        """
        _, _, stress = self.prop.eval(self.__unitcell_dict)
        self.eq_stress = -np.array(
            [
                [stress[0], stress[3], stress[5]],
                [stress[3], stress[1], stress[4]],
                [stress[5], stress[4], stress[2]],
            ]
        )

    def __run_initial_geometry_optimization(self):

        print("-------------------------------")
        print("Running geometry optimization")
        minobj = MinimizeSym(
            self.__unitcell_dict, properties=self.prop, relax_cell=True
        )
        minobj.run(gtol=1e-6)
        res_f, res_s = minobj.residual_forces
        print("Residuals (force):")
        print(res_f.T)
        print("Residuals (stress):")
        print(res_s)
        print("E0:", minobj.energy)
        print("n_iter:", minobj.n_iter)
        print("Success:", minobj.success)
        print("-------------------------------")

        if minobj.success:
            self.__unitcell_dict = minobj.structure

    def __run_geometry_optimization(self, st_dict):

        print("Running geometry optimization")
        try:
            minobj = MinimizeSym(st_dict, properties=self.prop, relax_cell=False)
            minobj.run(gtol=1e-6)
            print("Success:", minobj.success)
            if minobj.success:
                st_dict = minobj.structure
        except ValueError:
            print("No degrees of freedom in geometry optimization")

        return st_dict

    def run(self):

        deform = DeformedStructureSet(self.st_pmg)
        strains = [d.green_lagrange_strain for d in deform.deformations]

        st_dict_deform = []
        for i in range(len(deform)):
            st_dict = copy.deepcopy(self.__unitcell_dict)
            lattice = np.array(deform[i].as_dict()["lattice"]["matrix"])
            st_dict["axis"] = lattice.T

            # st_dict = self.__run_geometry_optimization(st_dict)
            st_dict_deform.append(st_dict)

        _, _, stresses = self.prop.eval_multiple(st_dict_deform)
        stresses = convert_stresses_in_gpa(stresses, st_dict_deform)
        stresses = -np.array(
            [
                [[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]]
                for s in stresses
            ]
        )

        const = diff_fit(strains, stresses, eq_stress=self.eq_stress)[0]

        ids = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        self.__elastic_constants = np.array(
            [[const[i1[0]][i1[1]][i2[0]][i2[1]] for i2 in ids] for i1 in ids]
        )

        return self

    def write_elastic_constants(self, filename="polymlp_elastic.yaml"):

        f = open(filename, "w")

        print("elastic_constants:", file=f)
        print("  unit: GPa", file=f)

        ids = [
            (1, 1),
            (2, 2),
            (3, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 4),
            (3, 5),
            (3, 6),
            (4, 5),
            (4, 6),
            (5, 6),
        ]
        for i, j in ids:
            if self.__elastic_constants[i - 1][j - 1] > 1e-10:
                print(
                    "  c_" + str(i) + str(j) + ":",
                    "{:.6f}".format(self.__elastic_constants[i - 1][j - 1]),
                    file=f,
                )
            else:
                print("  c_" + str(i) + str(j) + ": 0", file=f)
        f.close()

    @property
    def elastic_constants(self):
        return self.__elastic_constants


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
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
    el = PolymlpElastic(unitcell, args.poscar, pot=args.pot)
    el.run()
    el.write_elastic_constants()
