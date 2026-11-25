"""Class for benckmark."""

import time
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Optional

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.supercell_utils import get_supercell


@dataclass
class TargetBenchmark:
    """Dataclass for benchmark target.

    Parameters
    ----------
    t_single: Computational time for single-core calculation in ms/atom/step.
    t_openmp: Computational time for OPENMP calculation in ms/atom/step.
    """

    poscar: PosixPath
    pot: PosixPath
    structure: PolymlpStructure
    prop: Properties
    t_single: Optional[float] = None
    t_openmp: Optional[float] = None


class PolymlpBenchmark:
    """Class for benckmark."""

    def __init__(self, verbose: bool = False, use_test: bool = False):
        """Init method."""
        self._verbose = verbose

        base_dir = Path(__file__).resolve().parent
        if use_test:
            max1 = 2
            max2 = 2
            max3 = 2
        else:
            max1 = 8
            max2 = 7
            max3 = 5

        poscars = [
            base_dir / "POSCAR.hcp.Ti",
            base_dir / "POSCAR.RS.MgO",
            base_dir / "POSCAR.perovskite.SrTiO3",
        ]
        pot_all = [
            [base_dir / f"polymlp.yaml.Ti.{i}" for i in range(1, max1)],
            [base_dir / f"polymlp.yaml.MgO.{i}" for i in range(1, max2)],
            [base_dir / f"polymlp.yaml.SrTiO3.{i}" for i in range(1, max3)],
        ]

        if self._verbose:
            print("Parsing structures and polymlps.")

        self._targets = []
        for poscar, p1 in zip(poscars, pot_all):
            st = Poscar(poscar).structure
            for p2 in p1:
                prop = Properties(pot=p2)
                self._targets.append(
                    TargetBenchmark(poscar=poscar, pot=p2, structure=st, prop=prop)
                )

    def run(self, n_calc: int = 20, size: tuple = (5, 5, 5)):
        """Estimate computational costs for polymlps."""
        for target in self._targets:
            target = self._run_single(target, n_calc=n_calc, size=size)
        return self._targets

    def _run_single(
        self,
        target: TargetBenchmark,
        n_calc: int = 20,
        size: tuple = (5, 5, 5),
    ):
        """Estimate computational costs for single polymlp."""
        supercell = get_supercell(target.structure, size)
        n_atom = len(supercell.elements)

        if self._verbose:
            print("-----", flush=True)
            print("polymlp:", target.pot.name, flush=True)

        t1 = time.perf_counter()
        _ = [target.prop.eval(supercell, use_openmp=False) for i in range(n_calc)]
        t2 = time.perf_counter()
        target.t_single = (t2 - t1) * 1000 / n_atom / n_calc
        if self._verbose:
            print("time_single:", target.t_single, "(ms/atom/step)", flush=True)

        structures = [supercell] * n_calc
        t1 = time.perf_counter()
        _ = target.prop.eval_multiple(structures)
        t2 = time.perf_counter()
        target.t_openmp = (t2 - t1) * 1000 / n_atom / n_calc
        if self._verbose:
            print("time_openmp:", target.t_openmp, "(ms/atom/step)", flush=True)

        return target

    def save(self, filename: str = "polymlp_benchmark.yaml"):
        """Save benchmark."""
        with open(filename, "w") as f:
            print("unit: ms/atom/step", file=f)
            print("benchmark:", file=f)
            for target in self._targets:
                print("- pot:     ", target.pot.name, file=f)
                print("  t_single:", target.t_single, file=f)
                print("  t_openmp:", target.t_openmp, file=f)
                print(file=f)
