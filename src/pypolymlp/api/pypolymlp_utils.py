"""API Class for using utility functions."""

from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.count_time import PolymlpCost
from pypolymlp.utils.dataset.auto_divide import auto_divide
from pypolymlp.utils.grid_search.optimal import find_optimal_mlps
from pypolymlp.utils.vasp_utils import (
    load_electronic_properties_from_vasprun,
    print_poscar,
    write_poscar_file,
)
from pypolymlp.utils.vasprun_compress import compress_vaspruns


class PypolymlpUtils:
    """API Class for using utility functions."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._structure = None
        self._sym = None

    def compress_vaspruns(self, vaspruns: list[str], n_jobs: int = 1):
        """Compress vasprun.xml files.

        Parameters
        ----------
        vaspruns: vasprun.xml files
        n_jobs: Number of jobs performed simultaneously.
        """
        if n_jobs == 1:
            for vasp in vaspruns:
                compress_vaspruns(vasp)
        else:
            from joblib import Parallel, delayed

            _ = Parallel(n_jobs=n_jobs)(
                delayed(compress_vaspruns)(vasp) for vasp in vaspruns
            )
        return self

    def compute_electron_properties_from_vaspruns(
        self,
        vaspruns: list[str],
        temp_max: float = 2000,
        temp_step: float = 50,
        n_jobs: int = 1,
    ):
        """Compute finite-temperature electronic properties.

        Parameters
        ----------
        vaspruns: vasprun.xml files
        temp_max: Maximum temperature.
        temp_step: Interval of temperatures.
        n_jobs: Number of jobs performed simultaneously.
        """
        output_files = []
        for vasp in vaspruns:
            split = vasp.split("/")
            output = (
                "/".join(split[:-1])
                + "/electron"
                + split[-1].replace("vasprun", "").replace("xml", "yaml")
            )
            output_files.append(output)
        if n_jobs == 1:
            for vasp, output in zip(vaspruns, output_files):
                load_electronic_properties_from_vasprun(
                    vasp,
                    output_filename=output,
                    temp_max=temp_max,
                    temp_step=temp_step,
                )
        else:
            from joblib import Parallel, delayed

            _ = Parallel(n_jobs=n_jobs)(
                delayed(load_electronic_properties_from_vasprun)(
                    vasp,
                    output_filename=output,
                    temp_max=temp_max,
                    temp_step=temp_step,
                )
                for vasp, output in zip(vaspruns, output_files)
            )

    def estimate_polymlp_comp_cost(
        self,
        pot: Optional[Union[str, list[str]]] = None,
        path_pot: Optional[list] = None,
        poscar: Optional[str] = None,
        supercell: np.ndarray = (4, 4, 4),
        n_calc: int = 50,
    ):
        """Estimate computational cost of MLP.

        Parameters
        ----------
        pot: Polynomial MLP file or files for hybrid models
        path_pot: List of directory paths for polymlps.
                  (e.g. [grid/polymlp-00001, grid/polymlp-00002, ...])
        poscar: Structure in POSCAR format used for calculating properties.
        supercell: Diagonal supercell size.
        n_calc: Number of single-point calculations.
        """
        pycost = PolymlpCost(
            pot=pot,
            path_pot=path_pot,
            poscar=poscar,
            supercell=supercell,
            verbose=self._verbose,
        )
        pycost.run(n_calc=n_calc)

    def find_optimal_mlps(
        self,
        dirs: list,
        key: str,
        use_force: bool = False,
        use_logscale_time: bool = False,
    ):
        """Find optimal MLPs on the convex hull.

        Parameters
        ----------
        dirs: Multiple directories containing
              (polymlp_costs.yaml, polymlp_error.yaml) files.
        key: Key used for defining MLP accuracy.
             RMSE for a dataset containing the key in polymlp.error.yaml is used.
        use_force: Use errors for forces to define MLP accuracy.
        use_logscale_time: Use time in log scale to define MLP efficiency.
        """
        find_optimal_mlps(dirs, key)

    def divide_dataset(self, vaspruns: list[str]):
        """Divide a dataset into training and test datasets automatically.

        Generate divided subsets and texts that will be included in input files.

        Parameters
        ----------
        vaspruns: vasprun.xml files
        """
        auto_divide(vaspruns, verbose=self._verbose)

    def init_symmetry(
        self,
        structure: Optional[PolymlpStructure] = None,
        poscar: Optional[str] = None,
        symprec: float = 1e-4,
    ):
        """Initialize spglib instance."""
        from pypolymlp.utils.spglib_utils import SymCell

        self._sym = SymCell(poscar_name=poscar, st=structure, symprec=symprec)
        return self

    def refine_cell(self):
        """Refine structure using a procedure implemented in spglib."""
        structure = self._sym.refine_cell()
        return structure

    def get_spacegroup(self):
        """Retrun space group of structure."""
        return self._sym.get_spacegroup()

    def print_poscar(self, structure: PolymlpStructure):
        """Print structure in poscar format."""
        print_poscar(structure)

    def write_poscar_file(
        self,
        structure: PolymlpStructure,
        filename: str = "poscar_pypolymlp",
    ):
        """Save structure in poscar format."""
        write_poscar_file(structure, filename=filename)
