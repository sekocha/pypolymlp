"""Class for parsing prediction data and generating web contents."""

import glob
import os
import shutil

# import subprocess
import tarfile
from datetime import datetime

#
# import numpy as np
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

    def _find_active_mlps(self):
        """Find active MLPs."""
        min_rmse = min([float(d["rmse_energy"]) for d in self._polymlps])
        threshold = min(10, min_rmse * 2) if min_rmse > 2.5 else 5.0
        for i, d in enumerate(self._polymlps):
            d["active"] = float(d["rmse_energy"]) < threshold
        return self

        # yamlfile = path_data + "/polymlp_summary/prediction.yaml"
        # yamldata = self.__read_yaml(yamlfile)
        # self.__structures = yamldata["structures"]

    def run(self):
        """Generate all contents for repository."""
        self.compress_mlps()
        self.generate_summary()
        self.generate_predictions()

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

        generate_summary_txt(self._path_web, self._polymlps_id, self._polymlps)
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


#     def __get_num_structures(self):
#
#         file1 = "/".join([self.__polymlps[0]["id"], "energy_dist", "energy-train.dat"])
#         cmd = "wc -l " + file1
#         c1 = subprocess.check_output(cmd.split()).decode().split()[0]
#         file2 = "/".join([self.__polymlps[0]["id"], "energy_dist", "energy-test.dat"])
#         cmd = "wc -l " + file2
#         c2 = subprocess.check_output(cmd.split()).decode().split()[0]
#         return int(c1) + int(c2) - 2
#

#     def __read_lattice_constants_yaml(self, yamlfile):
#
#         C = self.__read_yaml(yamlfile)["standardized_lattice_constants"]
#         data_output = [
#             C["volume"],
#             C["a"],
#             C["b"],
#             C["c"],
#             C["alpha"],
#             C["beta"],
#             C["gamma"],
#         ]
#         data_output = np.round(np.array(data_output).astype(float), decimals=4)
#         return data_output
#
#     def __read_elastic_yaml(self, yamlfile):
#
#         C = self.__read_yaml(yamlfile)["elastic_constants"]
#         data_output = [
#             [
#                 C["c_11"],
#                 C["c_12"],
#                 C["c_13"],
#                 C["c_14"],
#                 C["c_15"],
#                 C["c_16"],
#             ],
#             [
#                 C["c_12"],
#                 C["c_22"],
#                 C["c_23"],
#                 C["c_24"],
#                 C["c_25"],
#                 C["c_26"],
#             ],
#             [
#                 C["c_13"],
#                 C["c_23"],
#                 C["c_33"],
#                 C["c_34"],
#                 C["c_35"],
#                 C["c_36"],
#             ],
#             [
#                 C["c_14"],
#                 C["c_24"],
#                 C["c_34"],
#                 C["c_44"],
#                 C["c_45"],
#                 C["c_46"],
#             ],
#             [
#                 C["c_15"],
#                 C["c_25"],
#                 C["c_35"],
#                 C["c_45"],
#                 C["c_55"],
#                 C["c_56"],
#             ],
#             [
#                 C["c_16"],
#                 C["c_26"],
#                 C["c_36"],
#                 C["c_46"],
#                 C["c_56"],
#                 C["c_66"],
#             ],
#         ]
#         data_output = np.round(np.array(data_output).astype(float), decimals=1)
#         data_output[np.where(np.abs(data_output) < 1e-5)] = 0
#         return data_output
#
#     def __read_icsd_yaml(self, yamlfile):
#
#         data = self.__read_yaml(yamlfile)["icsd_predictions"]
#
#         sort_ids = np.argsort([float(d["mlp"]) for d in data])
#         data = [
#             [
#                 data[i]["prototype"],
#                 round(float(data[i]["mlp"]), 5),
#                 round(float(data[i]["dft"]), 5),
#             ]
#             for i in sort_ids
#         ]
#         return np.array(data)
#
#     def run_predictions(self):
#
#         for d in self.__polymlps:
#             if d["distribution"]:
#                 path_data = "/".join([self.__path_data, d["id"], "predictions"]) + "/"
#                 path_output = (
#                     "/".join([self.__path_output, "predictions", d["id"]]) + "/"
#                 )
#                 os.makedirs(path_output, exist_ok=True)
#                 f = open(path_output + "prediction.rst", "w")
#                 print(":orphan:", file=f)
#                 print("", file=f)
#                 print(
#                     "----------------------------------------------------",
#                     file=f,
#                 )
#                 print(d["id"] + " (" + self.__polymlps_id + ")", file=f)
#                 print(
#                     "----------------------------------------------------",
#                     file=f,
#                 )
#                 print("", file=f)
#
#                 path_dist = "/".join([self.__path_data, d["id"], "energy_dist"]) + "/"
#                 file_copied = path_dist + "distribution.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "distribution.png",
#                         height=600,
#                         file=f,
#                         title="Energy distribution",
#                     )
#
#                 file_copied = path_data + "polymlp_icsd_pred.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "polymlp_icsd_pred.png",
#                         height=300,
#                         file=f,
#                         title="Prototype structure energy",
#                     )
#
#                 file_copied = path_data + "polymlp_eos.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "polymlp_eos.png",
#                         height=350,
#                         file=f,
#                         title="Equation of state",
#                     )
#
#                 file_copied = path_data + "polymlp_eos_sep.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "polymlp_eos_sep.png",
#                         height=700,
#                         file=f,
#                         title="Equation of state for each structure",
#                     )
#
#                 file_copied = path_data + "polymlp_phonon_dos.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "polymlp_phonon_dos.png",
#                         height=600,
#                         file=f,
#                         title="Phonon density of states",
#                     )
#
#                 file_copied = path_data + "polymlp_thermal_expansion.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "polymlp_thermal_expansion.png",
#                         height=400,
#                         file=f,
#                         title="Thermal expansion",
#                     )
#
#                 file_copied = path_data + "polymlp_bulk_modulus.png"
#                 if os.path.exists(file_copied):
#                     self.__copy(file_copied, path_output)
#                     include_image(
#                         "polymlp_bulk_modulus.png",
#                         height=400,
#                         file=f,
#                         title="Bulk modulus (temperature dependence)",
#                     )
#
#                 print("**Prototype structure energy**", file=f)
#                 print("", file=f)
#                 yamlfile = "/".join([path_data, "polymlp_icsd_pred.yaml"])
#                 icsd_data = self.__read_icsd_yaml(yamlfile)
#                 text_to_csv_table(
#                     icsd_data,
#                     title="Prototype structure energy",
#                     header="Prototype, MLP (eV/atom), DFT (eV/atom)",
#                     widths=[25, 10, 10],
#                     file=f,
#                 )
#
#                 print("**Lattice constants**", file=f)
#                 print("", file=f)
#                 for st in self.__structures:
#                     yamlfile = "/".join(
#                         [
#                             path_data,
#                             st["st_type"],
#                             "polymlp_lattice_constants.yaml",
#                         ]
#                     )
#                     if os.path.exists(yamlfile):
#                         lc_data = self.__read_lattice_constants_yaml(yamlfile)
#                         text_to_csv_table(
#                             [lc_data],
#                             title=("Lattice constants [" + st["st_type"] + "] (MLP)"),
#                             header=(
#                                 "volume (ang.^3), "
#                                 "a (ang.), b (ang.), c (ang.), "
#                                 "alpha, beta, gamma"
#                             ),
#                             widths=[12, 10, 10, 10, 8, 8, 8],
#                             file=f,
#                         )
#
#                     """
#                     yamlfile = '/'.join(
#                         [self.__path_data, 'vasp', st['st_type'],
#                         'polymlp_lattice_constants.yaml']
#                     )
#                     if os.path.exists(yamlfile):
#                         lc_data = self.__read_lattice_constants_yaml(yamlfile)
#                         text_to_csv_table(
#                             [lc_data],
#                             title=('Lattice constants ['
#                                     + st['st_type'] + '] (DFT)'),
#                             header=('volume (ang.^3), '
#                                     'a (ang.), b (ang.), c (ang.), '
#                                     'alpha, beta, gamma'),
#                             widths=[12,10,10,10,8,8,8],
#                             file=f
#                         )
#                     """
#
#                 print("**Elastic constants**", file=f)
#                 print("", file=f)
#                 for st in self.__structures:
#                     yamlfile = "/".join(
#                         [path_data, st["st_type"], "polymlp_elastic.yaml"]
#                     )
#                     if os.path.exists(yamlfile):
#                         elastic_data = self.__read_elastic_yaml(yamlfile)
#                         text_to_csv_table(
#                             elastic_data,
#                             title="Elastic constants [" + st["st_type"] + "]",
#                             header="1, 2, 3, 4, 5, 6",
#                             widths=[10 for i in range(6)],
#                             file=f,
#                         )
#
#                 f.close()
#
