"""API Class for constructing repository entry."""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from pypolymlp.api.pypolymlp_utils import PypolymlpUtils
from pypolymlp.calculator.auto.pypolymlp_autocalc import PypolymlpAutoCalc
from pypolymlp.calculator.auto_repository.figures_summary import (
    plot_eqm_properties,
    plot_mlp_distribution,
)
from pypolymlp.calculator.auto_repository.web import WebContents
from pypolymlp.core.io_polymlp import convert_to_yaml, find_mlps


@dataclass
class MLPAttr:
    """Dataclass for attributes of polymlp."""

    entry_path: str
    mlp_id: str
    path_mlp: Optional[str] = None
    path_prediction: Optional[PypolymlpAutoCalc] = None
    autocalc: Optional[PypolymlpAutoCalc] = None

    def __post_init__(self):
        """Init method."""
        self.path_mlp = self.entry_path + "/polymlps/" + self.mlp_id
        self.path_prediction = self.entry_path + "/predictions/" + self.mlp_id
        os.makedirs(self.path_mlp, exist_ok=True)
        os.makedirs(self.path_prediction, exist_ok=True)

    def set_autocalc(self, verbose: bool = False):
        """Set autocalc instance."""
        if self.autocalc is None:
            pot = find_mlps(self.path_mlp)
            self.autocalc = PypolymlpAutoCalc(
                pot=pot,
                path_output=self.path_prediction,
                verbose=verbose,
            )
        return self


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
        self._utils = PypolymlpUtils(verbose=self._verbose)

        self._entry_path = None
        self._system = None
        self._convex_mlp_attrs = None
        self._times = None

        np.set_printoptions(legacy="1.21")

    def calc_costs(self, n_calc: int = 20):
        """Calculate computational costs of MLPs."""
        if self._verbose:
            print("Calculating computational costs of polymlps.", flush=True)
        if self._mlp_paths is None:
            raise RuntimeError("MLP paths not found.")

        self._utils.estimate_polymlp_comp_cost(
            path_pot=self._mlp_paths,
            n_calc=n_calc,
        )
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
        if self._verbose:
            print("Determining optimal polymlps on convex hull.", flush=True)
        if self._mlp_paths is None:
            raise RuntimeError("MLP paths not found.")

        summary_all, summary_convex, self._system = self._utils.find_optimal_mlps(
            self._mlp_paths,
            key,
            use_force=use_force,
            use_logscale_time=use_logscale_time,
        )
        datetime_str = datetime.now().strftime("%Y-%m-%d")
        self._entry_path = self._system + "-" + datetime_str + "/"
        self._times = summary_convex[:, 0].astype(float)
        self._copy_convex_mlp_files(self._system, abspaths=summary_convex[:, -1])

        path_summary = self._entry_path + "/summary"
        plot_mlp_distribution(
            summary_all,
            summary_convex,
            self._system,
            path_output=path_summary,
        )
        return self

    def _copy_convex_mlp_files(self, system: str, abspaths: list):
        """Copy files for convex MLPs."""
        self._move_summary()

        self._convex_mlp_attrs = []
        for path in abspaths:
            mlp_id = path.split("/")[-1]
            mlp_attr = MLPAttr(entry_path=self._entry_path, mlp_id=mlp_id)
            self._copy_mlp(path, mlp_attr.path_mlp)
            for file in ("polymlp_cost.yaml", "polymlp_error.yaml"):
                shutil.copy(path + "/" + file, mlp_attr.path_mlp)

            self._convex_mlp_attrs.append(mlp_attr)
        return self

    def _move_summary(self):
        """Move summary files."""
        path_summary = self._entry_path + "/summary"
        shutil.rmtree(path_summary, ignore_errors=True)
        os.makedirs(path_summary, exist_ok=True)

        shutil.move(
            "polymlp_summary_all.yaml",
            path_summary + "/polymlp_summary_all.yaml",
        )
        shutil.move(
            "polymlp_summary_convex.yaml",
            path_summary + "/polymlp_summary_convex.yaml",
        )
        return self

    def _copy_mlp(self, path_mlp_original: str, path_mlp_target: str):
        """Copy files of a single polymlp."""
        pot = find_mlps(path_mlp_original)
        legacy = convert_to_yaml(pot, yaml="polymlp.yaml")
        if legacy:
            for p in find_mlps("."):
                try:
                    shutil.move(p, path_mlp_target)
                except:
                    pass
                os.remove(p)
        else:
            for p in pot:
                try:
                    shutil.copy(p, path_mlp_target)
                except:
                    pass
        return self

    def calc_properties_elements(
        self,
        vaspruns_prototypes: Optional[list] = None,
        vaspruns_train: Optional[list] = None,
        vaspruns_test: Optional[list] = None,
        icsd_ids: Optional[list] = None,
    ):
        """Calculate properties."""
        if self._entry_path is None:
            raise RuntimeError("Run extract_convex_polymlps first.")
        if self._times is None:
            raise RuntimeError("Computational times not found.")
        if self._convex_mlp_attrs is None:
            raise RuntimeError("Convex MLPs not found.")

        prototypes_all = []
        for mlp_attr in self._convex_mlp_attrs:
            mlp_attr.set_autocalc(verbose=self._verbose)
            calc = mlp_attr.autocalc
            calc.run_prototypes()
            calc.save_prototypes()
            calc.plot_prototypes(self._system, mlp_attr.mlp_id)
            prototypes_all.append(calc.prototypes)

            if vaspruns_prototypes is not None:
                calc.calc_comparison_with_dft(
                    vaspruns=vaspruns_prototypes,
                    icsd_ids=icsd_ids,
                )
                calc.plot_comparison_with_dft(self._system, mlp_attr.mlp_id)

            if vaspruns_train is not None and vaspruns_test is not None:
                calc.calc_energy_distribution(vaspruns_train, vaspruns_test)
                calc.plot_energy_distribution(self._system, mlp_attr.mlp_id)

        plot_eqm_properties(
            prototypes_all,
            self._times,
            self._system,
            path_output=self._entry_path + "/predictions",
        )
        return self

    def calc_properties_binary_alloys(
        self,
        vaspruns_binary_prototypes: Optional[list] = None,
        vaspruns_element_prototypes1: Optional[list] = None,
        vaspruns_element_prototypes2: Optional[list] = None,
        icsd_ids_binary: Optional[list] = None,
        icsd_ids_element1: Optional[list] = None,
        icsd_ids_element2: Optional[list] = None,
        vaspruns_train: Optional[list] = None,
        vaspruns_test: Optional[list] = None,
    ):
        """Calculate properties."""
        if self._entry_path is None:
            raise RuntimeError("Run extract_convex_polymlps first.")
        if self._times is None:
            raise RuntimeError("Computational times not found.")
        if self._convex_mlp_attrs is None:
            raise RuntimeError("Convex MLPs not found.")

        elements = self._system.split("-")
        prototypes_all = []
        for mlp_attr in self._convex_mlp_attrs:
            mlp_attr.set_autocalc(verbose=self._verbose)
            calc = mlp_attr.autocalc
            calc.run_prototypes()
            calc.save_prototypes()
            calc.plot_prototypes(self._system, mlp_attr.mlp_id)
            prototypes_all.append(calc.prototypes)

        if vaspruns_binary_prototypes is not None:
            for mlp_attr in self._convex_mlp_attrs:
                calc = mlp_attr.autocalc
                calc.calc_comparison_with_dft(
                    vaspruns=vaspruns_binary_prototypes,
                    icsd_ids=icsd_ids_binary,
                )
                calc.plot_comparison_with_dft(
                    self._system,
                    mlp_attr.mlp_id,
                    filename_suffix=self._system,
                )

        if vaspruns_element_prototypes1 is not None:
            for mlp_attr in self._convex_mlp_attrs:
                calc = mlp_attr.autocalc
                calc.calc_comparison_with_dft(
                    vaspruns=vaspruns_element_prototypes1,
                    icsd_ids=icsd_ids_element1,
                )
                calc.plot_comparison_with_dft(
                    elements[0] + " in " + self._system,
                    mlp_attr.mlp_id,
                    filename_suffix=elements[0],
                )
        if vaspruns_element_prototypes2 is not None:
            for mlp_attr in self._convex_mlp_attrs:
                calc = mlp_attr.autocalc
                calc.calc_comparison_with_dft(
                    vaspruns=vaspruns_element_prototypes2,
                    icsd_ids=icsd_ids_element2,
                )
                calc.plot_comparison_with_dft(
                    elements[1] + " in " + self._system,
                    mlp_attr.mlp_id,
                    filename_suffix=elements[1],
                )

        if vaspruns_binary_prototypes is not None:
            vaspruns = list(vaspruns_binary_prototypes)
            icsd_ids = list(icsd_ids_binary)
            if vaspruns_element_prototypes1 is not None:
                vaspruns.extend(vaspruns_element_prototypes1)
                icsd_ids.extend(icsd_ids_element1)
            if vaspruns_element_prototypes2 is not None:
                vaspruns.extend(vaspruns_element_prototypes2)
                icsd_ids.extend(icsd_ids_element2)

            for mlp_attr in self._convex_mlp_attrs:
                calc = mlp_attr.autocalc
                calc.calc_formation_energies(
                    vaspruns=vaspruns,
                    icsd_ids=icsd_ids,
                )
                calc.plot_binary_formation_energies(self._system, mlp_attr.mlp_id)

        if vaspruns_train is not None and vaspruns_test is not None:
            for mlp_attr in self._convex_mlp_attrs:
                calc = mlp_attr.autocalc
                calc.calc_energy_distribution(vaspruns_train, vaspruns_test)
                calc.plot_energy_distribution(self._system, mlp_attr.mlp_id)

        for i, comp_range in enumerate(
            [(-0.01, 0.01), (0.01, 0.45), (0.45, 0.55), (0.55, 0.99), (0.99, 1.01)]
        ):
            prototypes = []
            for p1 in prototypes_all:
                p_each_mlp = []
                for p2 in p1:
                    if p2.structure_eq is None:
                        continue
                    n_atoms = p2.structure_eq.n_atoms
                    st_elements = p2.structure_eq.elements
                    if len(n_atoms) == 1 and st_elements[0] == elements[0]:
                        comp = 0.0
                    elif len(n_atoms) == 1 and st_elements[0] == elements[1]:
                        comp = 1.0
                    else:
                        comp = n_atoms[1] / sum(n_atoms)
                    if comp > comp_range[0] and comp < comp_range[1]:
                        p_each_mlp.append(p2)
            prototypes.append(p_each_mlp)

            plot_eqm_properties(
                prototypes,
                self._times,
                self._system,
                path_output=self._entry_path + "/predictions",
                filename_suffix="c" + str(i),
            )
        return self

    def generate_web_contents(self, path_prediction: Optional[str] = None):
        """Generate web contents."""
        if path_prediction is None:
            path_prediction = self._entry_path

        web = WebContents(path_prediction=path_prediction)
        web.run()
        return self

    @property
    def mlp_paths(self):
        """Return paths of polymlp locations."""
        return self._mlp_paths

    @mlp_paths.setter
    def mlp_paths(self, paths: list):
        """Set paths of polymlp locations."""
        self._mlp_paths = paths
