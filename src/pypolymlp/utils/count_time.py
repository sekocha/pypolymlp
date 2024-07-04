#!/usr/bin/env python
import argparse
import glob
import time

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import Poscar


class PolymlpCost:

    def __init__(self, pot_path=None, pot=None, poscar=None, supercell=[4, 4, 4]):

        self.pot_path = pot_path
        self.pot = pot

        if pot_path is None:
            prop = Properties(pot=pot)
        else:
            pot = sorted(glob.glob(pot_path[0] + "/polymlp.lammps*"))
            prop = Properties(pot=pot)

        if isinstance(prop.params_dict, list):
            self.elements = prop.params_dict[0]["elements"]
        else:
            self.elements = prop.params_dict["elements"]

        self.__set_structure(poscar=poscar)

    def __set_structure(self, poscar=None, supercell=[4, 4, 4]):
        from phonopy import Phonopy

        from pypolymlp.utils.phonopy_utils import (
            phonopy_cell_to_st_dict,
            st_dict_to_phonopy_cell,
        )

        if poscar is not None:
            unitcell_dict = Poscar(poscar).get_structure()
        else:
            unitcell_dict = dict()
            if len(self.elements) == 1:
                unitcell_dict["axis"] = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
                unitcell_dict["positions"] = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ]
                ).T
                unitcell_dict["n_atoms"] = np.array([4])
                unitcell_dict["types"] = np.array([0, 0, 0, 0])
                unitcell_dict["elements"] = [
                    self.elements[t] for t in unitcell_dict["types"]
                ]
                unitcell_dict["volume"] = np.linalg.det(unitcell_dict["axis"])
            elif len(self.elements) == 2:
                unitcell_dict["axis"] = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
                unitcell_dict["positions"] = np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ]
                ).T
                unitcell_dict["n_atoms"] = np.array([2, 2])
                unitcell_dict["types"] = np.array([0, 0, 1, 1])
                unitcell_dict["elements"] = [
                    self.elements[t] for t in unitcell_dict["types"]
                ]
                unitcell_dict["volume"] = np.linalg.det(unitcell_dict["axis"])
            else:
                raise ValueError("No structure setting for " "more than binary system")

        supercell_matrix = np.diag(supercell)
        unitcell = st_dict_to_phonopy_cell(unitcell_dict)
        phonopy = Phonopy(unitcell, supercell_matrix)
        self.supercell_dict = phonopy_cell_to_st_dict(phonopy.supercell)

    def run_single(self, pot, n_calc=20):

        prop = Properties(pot=pot)
        print("Calculations have been started.")
        t1 = time.time()
        for i in range(n_calc):
            e, _, _ = prop.eval(self.supercell_dict)
        t2 = time.time()

        n_atoms_sum = sum(self.supercell_dict["n_atoms"])
        cost1 = (t2 - t1) / n_atoms_sum / n_calc
        cost1 *= 1000
        print("Total time (sec):", t2 - t1)
        print("Number of atoms:", n_atoms_sum)
        print("Number of steps:", n_calc)
        print("Computational cost (msec/atom/step):", cost1)

        print("Calculations have been started (openmp).")
        n_calc2 = n_calc * 10
        st_dicts = [self.supercell_dict for i in range(n_calc2)]

        t3 = time.time()
        _, _, _ = prop.eval_multiple(st_dicts)
        t4 = time.time()

        cost2 = (t4 - t3) / n_atoms_sum / n_calc2
        cost2 *= 1000
        print("Total time (sec):", t4 - t3)
        print("Number of atoms:", n_atoms_sum)
        print("Number of steps:", n_calc2)
        print("Computational cost (msec/atom/step):", cost2)

        return cost1, cost2

    def run(self, n_calc=20):

        if self.pot_path is None:
            cost1, cost2 = self.run_single(self.pot, n_calc=n_calc)
            self.write_single_yaml(cost1, cost2, filename="polymlp_cost.yaml")
        else:
            # pot_dirs = sorted(glob.glob(self.pot_path + '/*'))
            pot_dirs = sorted(self.pot_path)
            for dir1 in pot_dirs:
                print("------- Target MLP:", dir1, "-------")
                pot = sorted(glob.glob(dir1 + "/polymlp.lammps*"))
                cost1, cost2 = self.run_single(pot, n_calc=n_calc)
                self.write_single_yaml(
                    cost1, cost2, filename=dir1 + "/polymlp_cost.yaml"
                )

    def write_single_yaml(self, cost1, cost2, filename="polymlp_cost.yaml"):

        f = open(filename, "w")
        print("units:", file=f)
        print("  time: msec/atom/step", file=f)
        print("", file=f)
        print("costs:", file=f)
        print("  single_core:", cost1, file=f)
        print("  openmp:     ", cost2, file=f)
        f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", type=str, default=None, help="poscar file")
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.lammps",
        help="polymlp file",
    )

    parser.add_argument(
        "-d",
        "--dirs",
        nargs="*",
        type=str,
        default=None,
        help="directory path",
    )

    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=[4, 4, 4],
        help="supercell size",
    )
    parser.add_argument("--n_calc", type=int, default=20, help="number of calculations")
    args = parser.parse_args()

    pycost = PolymlpCost(
        pot_path=args.dirs,
        pot=args.pot,
        poscar=args.poscar,
        supercell=args.supercell,
    )
    pycost.run(n_calc=args.n_calc)
