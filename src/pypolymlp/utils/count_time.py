"""Class for estimating computational cost of polymlp."""

import time
from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import find_mlps, load_mlps
from pypolymlp.utils.structure_utils import supercell_diagonal


class PolymlpCost:
    """Class for estimating computational cost of polymlp."""

    def __init__(
        self,
        pot: Optional[Union[str, list[str]]] = None,
        path_pot: Optional[str] = None,
        poscar: Optional[str] = None,
        supercell: np.ndarray = np.array([4, 4, 4]),
        verbose: bool = False,
    ):
        """Init method."""
        self._pot = pot
        self._path_pot = path_pot
        self._poscar = poscar
        self._supercell_size = supercell
        self._verbose = verbose

        self._elements = self._parse_elements_from_pot()
        self._supercell = self._set_structure()

    def _parse_elements_from_pot(self):
        """Get elements from MLP file."""
        pot_elements = None
        if self._path_pot is None:
            if isinstance(self._pot, list):
                pot_elements = self._pot[0]
            else:
                pot_elements = self._pot
        else:
            for path in self._path_pot:
                pot_elements = find_mlps(path)[0]
                if pot_elements is not None:
                    break

        if pot_elements is None:
            raise RuntimeError("polymlp potential files not found.")

        params, _ = load_mlps(pot_elements)
        self._elements = params.elements
        self._system = "-".join(self._elements)
        return self._elements

    def _set_structure(self):
        """Set a structure to calculate properties."""
        if self._poscar is not None:
            unitcell = Poscar(self._poscar).structure
        else:
            axis = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
            positions = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ]
            ).T
            if len(self._elements) == 1:
                n_atoms = np.array([4])
                types = np.array([0, 0, 0, 0])
            elif len(self._elements) == 2:
                n_atoms = np.array([2, 2])
                types = np.array([0, 0, 1, 1])
            elif len(self._elements) == 3:
                n_atoms = np.array([1, 1, 2])
                types = np.array([0, 1, 2, 2])
            else:
                raise RuntimeError("No structure setting for more than ternary system.")

            elements = [self._elements[t] for t in types]
            volume = np.linalg.det(axis)
            unitcell = PolymlpStructure(
                axis,
                positions,
                n_atoms,
                elements,
                types,
                volume,
            )

        self._supercell = supercell_diagonal(unitcell, size=self._supercell_size)
        return self._supercell

    def _run_single(self, pot: Union[str, list[str]], n_calc: int = 20):
        """Estimate computational cost for a single polymlp."""
        if self._verbose:
            print("Calculations have been started (openmp).")

        prop = Properties(pot=pot)

        n_atoms_sum = sum(self._supercell.n_atoms)
        n_calc2 = n_calc * 10
        structures = [self._supercell for i in range(n_calc2)]
        t3 = time.time()
        _, _, _ = prop.eval_multiple(structures)
        t4 = time.time()
        cost2 = (t4 - t3) * 1000 / n_atoms_sum / n_calc2

        if self._verbose:
            print("Total time (sec):", t4 - t3)
            print("Number of atoms:", n_atoms_sum)
            print("Number of steps:", n_calc2)
            print("Computational cost (msec/atom/step):", cost2)

        if self._verbose:
            print("Calculations have been started.")

        t1 = time.time()
        _ = [prop.eval(self._supercell, use_openmp=False) for i in range(n_calc)]
        t2 = time.time()
        cost1 = (t2 - t1) * 1000 / n_atoms_sum / n_calc

        if self._verbose:
            print("Total time (sec):", t2 - t1)
            print("Number of atoms:", n_atoms_sum)
            print("Number of steps:", n_calc)
            print("Computational cost (msec/atom/step):", cost1)

        return cost1, cost2

    def run(self, n_calc: int = 20):
        """Estimate computational costs for polymlps."""
        if self._path_pot is None:
            cost1, cost2 = self._run_single(self._pot, n_calc=n_calc)
            self._write_single_yaml(cost1, cost2, filename="polymlp_cost.yaml")
        else:
            pot_dirs = sorted(self._path_pot)
            for dir1 in pot_dirs:
                pot = find_mlps(dir1)
                if self._verbose:
                    print("------- Target MLP:", dir1, "-------")
                    print("polymlp:", pot)

                if pot is not None:
                    cost1, cost2 = self._run_single(pot, n_calc=n_calc)
                    self._write_single_yaml(
                        cost1,
                        cost2,
                        filename=dir1 + "/polymlp_cost.yaml",
                    )

    def _write_single_yaml(
        self, cost1: float, cost2: float, filename: str = "polymlp_cost.yaml"
    ):
        """Save computational costs to a file."""
        f = open(filename, "w")
        print("system:", self._system, file=f)
        print("units:", file=f)
        print("  time: msec/atom/step", file=f)
        print("", file=f)
        print("costs:", file=f)
        print("  single_core:", cost1, file=f)
        print("  openmp:     ", cost2, file=f)
        f.close()
