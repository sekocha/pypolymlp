"""Class for parsing prediction data and generating web contents."""

import glob
import os
import shutil
import tarfile
from datetime import datetime

import yaml

from pypolymlp.calculator.auto.web_utils import (
    generate_predictions_txt,
    generate_summary_txt,
)
from pypolymlp.core.io_polymlp import find_mlps


class WebContents:
    """Class for parsing prediction data and generating web contents."""

    def __init__(
        self,
        path_prediction: str = "./",
        path_web: str = "./",
        verbose: bool = False,
    ):
        """Init method."""

        self._path_prediction = path_prediction
        self._verbose = verbose

        self._parse_optimal_mlps()
        self._set_paths(path_web)
        self._find_active_mlps()

    def _parse_optimal_mlps(self):
        """Parse optimal MLPs."""
        filename = self._path_prediction + "/summary/polymlp_summary_convex.yaml"
        data = yaml.safe_load(open(filename))
        self._system = data["system"]
        self._polymlps = data["polymlps"]
        self._elements = self._system.split("-")
        return self

    def _set_paths(self, path_web: str):
        """Set paths and potential ID."""
        today = datetime.now().strftime("%Y-%m-%d")
        self._polymlps_id = self._system + "-" + today
        for d in self._polymlps:
            if "hybrid" in d["id"]:
                self._polymlps_id += "-hybrid"
                break
        self._path_web = path_web + "/" + self._polymlps_id + "/"
        if self._verbose:
            path = os.path.abspath(self._path_web)
            print("Repository web contents are generated in", path, flush=True)
        return self

    def _find_active_mlps(self):
        """Find active MLPs."""
        min_rmse = min([float(d["rmse_energy"]) for d in self._polymlps])
        threshold = min(10, min_rmse * 2) if min_rmse > 2.5 else 5.0
        for i, d in enumerate(self._polymlps):
            d["active"] = float(d["rmse_energy"]) < threshold
        return self

    def run(self):
        """Generate all contents for repository."""
        self.compress_mlps()
        self.generate_summary()
        self.generate_predictions()
        return self

    def compress_mlps(self):
        """Compress MLP files."""
        path = self._path_web + "/polymlps/"
        os.makedirs(path, exist_ok=True)
        tar_all = tarfile.open(
            path + "polymlps-" + self._polymlps_id + ".tar.xz", "w:xz"
        )
        for d in self._polymlps:
            if not d["active"]:
                continue

            path_data = self._path_prediction + "/polymlps/" + d["id"] + "/"
            filename = path + d["id"] + ".tar.xz"
            tar = tarfile.open(filename, "w:xz")
            for name in find_mlps(path_data):
                tar.add(name, arcname=name.split("/")[-1])
            tar.close()
            tar_all.add(filename)
        tar_all.close()
        return self

    def generate_summary(self):
        """Generate contents for summary."""
        path = self._path_web + "/summary/"
        os.makedirs(path, exist_ok=True)
        files = [
            self._path_prediction + "/summary/polymlp_convex.png",
            self._path_prediction + "/predictions/polymlp_eqm_properties.png",
        ]
        for f in files:
            shutil.copy(f, path)

        generate_summary_txt(
            self._path_web,
            self._path_prediction,
            self._polymlps_id,
            self._polymlps,
        )
        return self

    def generate_predictions(self):
        """Generate contents for predictions."""
        for d in self._polymlps:
            if not d["active"]:
                continue
            path = self._path_web + "/predictions/" + d["id"] + "/"
            os.makedirs(path, exist_ok=True)
            path_prediction = self._path_prediction + "/predictions/" + d["id"] + "/"
            files = glob.glob(path_prediction + "/polymlp_*.png")
            for f in files:
                shutil.copy(f, path)

        generate_predictions_txt(
            self._path_web,
            self._path_prediction,
            self._polymlps_id,
            self._polymlps,
        )
        return self
