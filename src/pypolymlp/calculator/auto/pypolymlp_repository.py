"""API Class for constructing repository entry."""

import os
import shutil
from datetime import datetime
from typing import Optional

import numpy as np

from pypolymlp.calculator.auto.figures_properties import (
    plot_eos,
    plot_eos_separate,
    plot_phonon,
    plot_qha,
)
from pypolymlp.calculator.auto.figures_summary import (
    plot_eqm_properties,
    plot_mlp_distribution,
)
from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.calculator.auto.web import WebContents
from pypolymlp.core.io_polymlp import find_mlps
from pypolymlp.postproc.count_time import PolymlpCost
from pypolymlp.utils.grid_search.optimal import find_optimal_mlps


class PypolymlpRepository:
    """API Class for constructing repository entry."""

    def __init__(self, mlp_paths: Optional[list[str]] = None, verbose: bool = False):
        """Init method.

        Parameters
        ----------
        mlp_paths: Path of directory that contains MLPs from grid search.
        """
        self._mlp_paths = mlp_paths
        if self._mlp_paths is not None:
            if not isinstance(mlp_paths, (list, tuple, np.ndarray)):
                raise RuntimeError("mlp_paths must be array-like.")

        self._verbose = verbose

        self._entry_path = None
        self._system = None
        self._convex_mlp_paths = None
        self._times = None

        np.set_printoptions(legacy="1.21")

    def calc_costs(self, n_calc: int = 20):
        """Calculate computational costs of MLPs."""
        if self._mlp_paths is None:
            raise RuntimeError("MLP paths are required.")

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
        if self._mlp_paths is None:
            raise RuntimeError("MLP paths are required.")

        summary_all, summary_convex, self._system = find_optimal_mlps(
            self._mlp_paths,
            key,
            use_force=use_force,
            use_logscale_time=use_logscale_time,
            verbose=self._verbose,
        )
        self._times = summary_convex[:, 0].astype(float)
        abspaths = summary_convex[:, -1]
        self._copy_convex_mlp_files(self._system, abspaths)

        target = self._entry_path + "/summary"
        plot_mlp_distribution(
            summary_all,
            summary_convex,
            self._system,
            path_output=target,
        )
        return self

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
            target = self._entry_path + "/polymlps/" + name
            shutil.copytree(path, target)
            self._convex_mlp_paths.append(target)
        return self

    def calc_properties(
        self,
        vaspruns_prototypes: Optional[list] = None,
        vaspruns_train: Optional[list] = None,
        vaspruns_test: Optional[list] = None,
        icsd_ids: Optional[list] = None,
    ):
        """Calculate properties."""
        if self._entry_path is None:
            raise RuntimeError("Run extract_convex_polymlps first.")

        prototypes_all = []
        for path in self._convex_mlp_paths:
            name = path.split("/")[-1]
            target = self._entry_path + "/predictions/" + name
            calc = PypolymlpAutoCalc(
                pot=find_mlps(path),
                path_output=target,
                verbose=self._verbose,
            )
            calc.load_structures()
            calc.run()
            if vaspruns_prototypes is not None:
                calc.compare_with_dft(vaspruns=vaspruns_prototypes, icsd_ids=icsd_ids)
                calc.plot_comparison_with_dft(self._system, name)

            if vaspruns_train is not None and vaspruns_test is not None:
                calc.calc_energy_distribution(vaspruns_train, vaspruns_test)
                calc.plot_energy_distribution(self._system, name)

            calc.save_properties()
            prototypes_all.append(calc.prototypes)
            self._plot_properties(calc.prototypes, name, target)

        plot_eqm_properties(
            prototypes_all,
            self._times,
            self._system,
            path_output=self._entry_path + "/predictions",
        )
        return self

    def _plot_properties(self, prototypes: list, name: str, path_output: str):
        """Plot properties from single MLP."""
        n_type = len(self._system.split("-"))
        if n_type == 1:
            plot_eos(prototypes, self._system, name, path_output=path_output)

        plot_eos_separate(prototypes, self._system, name, path_output=path_output)
        plot_phonon(prototypes, self._system, name, path_output=path_output)
        plot_qha(
            prototypes,
            self._system,
            name,
            target="thermal_expansion",
            path_output=path_output,
        )
        plot_qha(
            prototypes,
            self._system,
            name,
            target="bulk_modulus",
            path_output=path_output,
        )
        return self

    def generate_web_contents(self, path_prediction: str = "./"):
        """Generate web contents."""
        web = WebContents(path_prediction=path_prediction)
        web.run()
