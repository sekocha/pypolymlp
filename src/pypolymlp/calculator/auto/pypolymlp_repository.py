"""API Class for constructing repository entry."""

import os
import shutil
from datetime import datetime

import numpy as np

from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.core.io_polymlp import find_mlps
from pypolymlp.postproc.count_time import PolymlpCost
from pypolymlp.utils.grid_search.optimal import find_optimal_mlps


class PypolymlpRepository:
    """API Class for constructing repository entry."""

    def __init__(
        self,
        mlp_paths: list[str],
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        mlp_paths: Path of directory that contains MLPs from grid search.
        """
        self._mlp_paths = mlp_paths
        if not isinstance(mlp_paths, (list, tuple, np.ndarray)):
            raise RuntimeError("mlp_paths must be array-like.")

        self._verbose = verbose

        self._entry_path = None
        self._convex_mlp_paths = None
        np.set_printoptions(legacy="1.21")

    def calc_costs(self, n_calc: int = 20):
        """Calculate computational costs of MLPs."""
        pycost = PolymlpCost(path_pot=self._mlp_paths, verbose=self._verbose)
        pycost.run(n_calc=n_calc)
        return self

    def extract_convex_polymlps(
        self,
        key: str,
        use_force: bool = False,
        use_logscale_time: bool = False,
    ):
        """Extract optimal MLPs on the convex hull.

        Parameters
        ----------
        key: Key used for defining MLP accuracy.
             RMSE for a dataset containing the key in polymlp.error.yaml is used.
        use_force: Use errors for forces to define MLP accuracy.
        use_logscale_time: Use time in log scale to define MLP efficiency.
        """
        system, abspaths = find_optimal_mlps(
            self._mlp_paths,
            key,
            use_force=use_force,
            use_logscale_time=use_logscale_time,
            verbose=self._verbose,
        )
        self._copy_convex_mlp_files(system, abspaths)

    def _copy_convex_mlp_files(self, system: str, abspaths: list):
        """Copy files for convex MLPs."""
        datetime_str = datetime.now().strftime("%Y-%m-%d")
        self._entry_path = system + "-" + datetime_str + "/"
        if os.path.exists(self._entry_path):
            os.remove("polymlp_summary_all.yaml")
            os.remove("polymlp_summary_convex.yaml")
            raise RuntimeError("Output directory already exists.")

        os.makedirs(self._entry_path + "/summary", exist_ok=True)
        shutil.move("polymlp_summary_all.yaml", self._entry_path + "/summary/")
        shutil.move("polymlp_summary_convex.yaml", self._entry_path + "/summary/")

        os.makedirs(self._entry_path + "/polymlps", exist_ok=True)
        self._convex_mlp_paths = []
        for path in abspaths:
            name = path.split("/")[-1]
            path_target = self._entry_path + "/polymlps/" + name
            shutil.copytree(path, path_target)
            self._convex_mlp_paths.append(path_target)
        return self

    def calc_properties(self):
        """Calculate properties."""
        if self._entry_path is None:
            raise RuntimeError("Run extract_convex_polymlps first.")

        prediction_path = self._entry_path + "/predictions/"
        for path in self._convex_mlp_paths:
            name = path.split("/")[-1]
            target = prediction_path + "/" + name + "/"
            os.makedirs(target, exist_ok=True)
            calc = PypolymlpAutoCalc(pot=find_mlps(path), verbose=self._verbose)
            calc.load_structures()
            calc.run(path_output=target)
