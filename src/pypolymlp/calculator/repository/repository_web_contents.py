#!/usr/bin/env python
import datetime
import glob
import os
import shutil
import subprocess
import tarfile

import numpy as np
import yaml

from pypolymlp.calculator.repository.utils.sphinx_utils import (
    include_image,
    text_to_csv_table,
)


class PolymlpRepositoryWebContents:

    def __init__(self, path_data="./"):

        self.__path_data = path_data
        yamlfile = path_data + "/polymlp_summary_convex.yaml"
        yamldata = self.__read_yaml(yamlfile)

        self.__system = yamldata["system"]
        self.__elements = self.__system.split("-")

        today = str(datetime.date.today())
        self.__polymlps_id = self.__system + "-" + today

        self.__polymlps = yamldata["polymlps"]
        for d in self.__polymlps:
            if "hybrid" in d["id"]:
                self.__polymlps_id += "-hybrid"
                break

        self.__path_output = path_data + "/" + self.__polymlps_id + "/"

        min_rmse = min([float(d["rmse_energy"]) for d in self.__polymlps])
        threshold = min(10, min_rmse * 2) if min_rmse > 2.5 else 5.0

        for d in self.__polymlps:
            if float(d["rmse_energy"]) < threshold:
                d["distribution"] = True
            else:
                d["distribution"] = False

        yamlfile = path_data + "/polymlp_summary/prediction.yaml"
        yamldata = self.__read_yaml(yamlfile)
        self.__structures = yamldata["structures"]

        print(
            "Repository web contents will be generated in",
            os.path.abspath(self.__path_output),
        )

        os.makedirs(self.__path_output + "/summary/", exist_ok=True)
        os.makedirs(self.__path_output + "/polymlps/", exist_ok=True)
        os.makedirs(self.__path_output + "/predictions/", exist_ok=True)

    def __read_yaml(self, yamlfile):

        f = open(yamlfile)
        yamldata = yaml.safe_load(f)
        f.close()
        return yamldata

    def __copy(self, file_copied, path_target):

        if os.path.exists(file_copied):
            shutil.copy(file_copied, path_target)
        return self

    def __get_num_structures(self):

        file1 = "/".join([self.__polymlps[0]["id"], "energy_dist", "energy-train.dat"])
        cmd = "wc -l " + file1
        c1 = subprocess.check_output(cmd.split()).decode().split()[0]
        file2 = "/".join([self.__polymlps[0]["id"], "energy_dist", "energy-test.dat"])
        cmd = "wc -l " + file2
        c2 = subprocess.check_output(cmd.split()).decode().split()[0]
        return int(c1) + int(c2) - 2

    def run_summary(self):

        path_output = self.__path_output + "/summary/"
        files = glob.glob(self.__path_data + "/polymlp_summary/*.png")
        for f1 in files:
            self.__copy(f1, path_output)

        f = open(self.__path_output + "/info.rst", "w")
        print(":orphan:", file=f)
        print("", file=f)
        print(self.__polymlps_id, file=f)
        print("====================================================", file=f)
        print("", file=f)

        include_image("summary/mlp_dist.png", title=None, height=350, file=f)
        # print('.. image:: summary/mlp_dist.png',file=f)
        # print('   :height: 350px', file=f)
        # print('', file=f)

        n_st = self.__get_num_structures()

        print(
            "The current structure dataset comprises "
            + str(n_st)
            + " structures. Procedures to generate structures"
            " and estimate MLPs are found in"
            " `A. Seko, J. Appl. Phys. 133, 011101 (2023)"
            " <https://doi.org/10.1063/5.0129045>`_.",
            file=f,
        )
        print("", file=f)

        include_image(
            "summary/eqm_properties.png",
            title="Predictions using Pareto optimal MLPs",
            height=350,
            file=f,
        )

        print(
            "These properties are calculated for MLP equilibrium structures"
            " obtained by performing local structure optimizations from"
            " the DFT equilibrium structures."
            " These DFT equilibrium structures are obtained by optimizing"
            " prototype structures that are included in ICSD."
            " As a result, the structure type of the converged structure may"
            " sometimes differ from the one shown in the legend.",
            file=f,
        )
        print("", file=f)
        print(
            "The other properties predicted using each Pareto optimal MLP"
            " are available from column **Predictions** "
            " in the following table.",
            file=f,
        )
        print("", file=f)

        min_rmse_f = min([float(d["rmse_force"]) for d in self.__polymlps])
        if min_rmse_f > 0.05:
            print(
                self.__polymlps_id,
                "shows large prediction errors."
                " Distributed MLPs should be carefully used."
                " MLPs are often accurate for reasonable structures,"
                " but it is sometimes inaccurate for unrealistic"
                " structures.",
                file=f,
            )
            print("", file=f)

        print("", file=f)
        print(".. csv-table:: Pareto optimals (on convex hull)", file=f)
        print(" :header: Name, Time, RMSE, Predictions, Files", file=f)
        print(" :widths: 15,8,8,6,15", file=f)
        print("", file=f)

        for d in self.__polymlps:
            id1 = d["id"]
            if d["distribution"]:
                if "hybrid" in id1:
                    polymlp_in_file = "polymlp.in.tar.gz"
                else:
                    polymlp_in_file = "polymlp.in"
                print(
                    " ",
                    id1,
                    ",",
                    "{:.3f}".format(float(d["cost_single"])),
                    "/",
                    "{:.3f}".format(float(d["cost_openmp"])),
                    ",",
                    "{:.3f}".format(float(d["rmse_energy"])),
                    "/",
                    "{:.4f}".format(float(d["rmse_force"])),
                    ",",
                    ":doc:`predictions <predictions/" + id1 + "/prediction>`,",
                    ":download:`polymlp.lammps <polymlps/"
                    + id1
                    + "/polymlp.lammps.tar.gz>`",
                    ":download:`polymlp.in <polymlps/"
                    + id1
                    + "/"
                    + polymlp_in_file
                    + ">`",
                    file=f,
                )
                # ':download:`log <polymlps/'+ id1 +'/log.dat>`'
            else:
                print(
                    " ",
                    id1,
                    ",",
                    "{:.3f}".format(float(d["cost_single"])),
                    "/",
                    "{:.3f}".format(float(d["cost_openmp"])),
                    ",",
                    "{:.3f}".format(float(d["rmse_energy"])),
                    "/",
                    "{:.4f}".format(float(d["rmse_force"])),
                    ",",
                    "--,--",
                    file=f,
                )

        print("", file=f)

        print("Units:", file=f)
        print("", file=f)
        print("* Time: [ms] (1core/36cores)", file=f)
        print("* RMSE: [meV/atom]/[eV/ang.]", file=f)
        print("", file=f)
        print(
            'Column "Time" shows the time required to compute the energy'
            " and forces for **1 MD step** and **1 atom**, which is"
            " estimated from 10 runs for a large structure using"
            " a workstation with Intel(R) Xeon(R) CPU E5-2695 v4 @ 2.10GHz.",
            file=f,
        )
        print(
            "Note that these MLPs should be carefully used for extreme"
            " structures. The MLPs often return meaningless values for them.",
            file=f,
        )
        print("", file=f)
        print(
            "- All Pareto optimal MLPs are available :download:`here"
            " <polymlps/polymlps-" + self.__polymlps_id + ".tar.gz>`.",
            file=f,
        )
        print("", file=f)

        f.close()
        return self

    def run_polymlps(self):

        for d in self.__polymlps:
            if d["distribution"]:
                path_data = "/".join([self.__path_data, d["id"], "polymlps"]) + "/"
                path_output = "/".join([self.__path_output, "polymlps", d["id"]]) + "/"
                os.makedirs(path_output, exist_ok=True)

                tar = tarfile.open(path_output + "polymlp.lammps.tar.gz", "w:gz")
                for name in glob.glob(path_data + "polymlp.lammps*"):
                    tar.add(name, arcname=name.split("/")[-1])
                tar.close()

                if "hybrid" in d["id"]:
                    tar = tarfile.open(path_output + "polymlp.in.tar.gz", "w:gz")
                    for name in glob.glob(path_data + "polymlp.in*"):
                        tar.add(name, arcname=name.split("/")[-1])
                    tar.close()
                else:
                    self.__copy(path_data + "polymlp.in", path_output)

        path_output = "/".join([self.__path_output, "polymlps"]) + "/"
        tar = tarfile.open(
            path_output + "polymlps-" + self.__polymlps_id + ".tar.gz",
            "w:gz",
        )
        tar.add(path_output)
        tar.close()

        return self

    def __read_lattice_constants_yaml(self, yamlfile):

        C = self.__read_yaml(yamlfile)["standardized_lattice_constants"]
        data_output = [
            C["volume"],
            C["a"],
            C["b"],
            C["c"],
            C["alpha"],
            C["beta"],
            C["gamma"],
        ]
        data_output = np.round(np.array(data_output).astype(float), decimals=4)
        return data_output

    def __read_elastic_yaml(self, yamlfile):

        C = self.__read_yaml(yamlfile)["elastic_constants"]
        data_output = [
            [
                C["c_11"],
                C["c_12"],
                C["c_13"],
                C["c_14"],
                C["c_15"],
                C["c_16"],
            ],
            [
                C["c_12"],
                C["c_22"],
                C["c_23"],
                C["c_24"],
                C["c_25"],
                C["c_26"],
            ],
            [
                C["c_13"],
                C["c_23"],
                C["c_33"],
                C["c_34"],
                C["c_35"],
                C["c_36"],
            ],
            [
                C["c_14"],
                C["c_24"],
                C["c_34"],
                C["c_44"],
                C["c_45"],
                C["c_46"],
            ],
            [
                C["c_15"],
                C["c_25"],
                C["c_35"],
                C["c_45"],
                C["c_55"],
                C["c_56"],
            ],
            [
                C["c_16"],
                C["c_26"],
                C["c_36"],
                C["c_46"],
                C["c_56"],
                C["c_66"],
            ],
        ]
        data_output = np.round(np.array(data_output).astype(float), decimals=1)
        data_output[np.where(np.abs(data_output) < 1e-5)] = 0
        return data_output

    def __read_icsd_yaml(self, yamlfile):

        data = self.__read_yaml(yamlfile)["icsd_predictions"]

        sort_ids = np.argsort([float(d["mlp"]) for d in data])
        data = [
            [
                data[i]["prototype"],
                round(float(data[i]["mlp"]), 5),
                round(float(data[i]["dft"]), 5),
            ]
            for i in sort_ids
        ]
        return np.array(data)

    def run_predictions(self):

        for d in self.__polymlps:
            if d["distribution"]:
                path_data = "/".join([self.__path_data, d["id"], "predictions"]) + "/"
                path_output = (
                    "/".join([self.__path_output, "predictions", d["id"]]) + "/"
                )
                os.makedirs(path_output, exist_ok=True)
                f = open(path_output + "prediction.rst", "w")
                print(":orphan:", file=f)
                print("", file=f)
                print(
                    "----------------------------------------------------",
                    file=f,
                )
                print(d["id"] + " (" + self.__polymlps_id + ")", file=f)
                print(
                    "----------------------------------------------------",
                    file=f,
                )
                print("", file=f)

                path_dist = "/".join([self.__path_data, d["id"], "energy_dist"]) + "/"
                file_copied = path_dist + "distribution.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "distribution.png",
                        height=600,
                        file=f,
                        title="Energy distribution",
                    )

                file_copied = path_data + "polymlp_icsd_pred.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "polymlp_icsd_pred.png",
                        height=300,
                        file=f,
                        title="Prototype structure energy",
                    )

                file_copied = path_data + "polymlp_eos.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "polymlp_eos.png",
                        height=350,
                        file=f,
                        title="Equation of state",
                    )

                file_copied = path_data + "polymlp_eos_sep.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "polymlp_eos_sep.png",
                        height=700,
                        file=f,
                        title="Equation of state for each structure",
                    )

                file_copied = path_data + "polymlp_phonon_dos.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "polymlp_phonon_dos.png",
                        height=600,
                        file=f,
                        title="Phonon density of states",
                    )

                file_copied = path_data + "polymlp_thermal_expansion.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "polymlp_thermal_expansion.png",
                        height=400,
                        file=f,
                        title="Thermal expansion",
                    )

                file_copied = path_data + "polymlp_bulk_modulus.png"
                if os.path.exists(file_copied):
                    self.__copy(file_copied, path_output)
                    include_image(
                        "polymlp_bulk_modulus.png",
                        height=400,
                        file=f,
                        title="Bulk modulus (temperature dependence)",
                    )

                print("**Prototype structure energy**", file=f)
                print("", file=f)
                yamlfile = "/".join([path_data, "polymlp_icsd_pred.yaml"])
                icsd_data = self.__read_icsd_yaml(yamlfile)
                text_to_csv_table(
                    icsd_data,
                    title="Prototype structure energy",
                    header="Prototype, MLP (eV/atom), DFT (eV/atom)",
                    widths=[25, 10, 10],
                    file=f,
                )

                print("**Lattice constants**", file=f)
                print("", file=f)
                for st in self.__structures:
                    yamlfile = "/".join(
                        [
                            path_data,
                            st["st_type"],
                            "polymlp_lattice_constants.yaml",
                        ]
                    )
                    if os.path.exists(yamlfile):
                        lc_data = self.__read_lattice_constants_yaml(yamlfile)
                        text_to_csv_table(
                            [lc_data],
                            title=("Lattice constants [" + st["st_type"] + "] (MLP)"),
                            header=(
                                "volume (ang.^3), "
                                "a (ang.), b (ang.), c (ang.), "
                                "alpha, beta, gamma"
                            ),
                            widths=[12, 10, 10, 10, 8, 8, 8],
                            file=f,
                        )

                    """
                    yamlfile = '/'.join(
                        [self.__path_data, 'vasp', st['st_type'],
                        'polymlp_lattice_constants.yaml']
                    )
                    if os.path.exists(yamlfile):
                        lc_data = self.__read_lattice_constants_yaml(yamlfile)
                        text_to_csv_table(
                            [lc_data],
                            title=('Lattice constants ['
                                    + st['st_type'] + '] (DFT)'),
                            header=('volume (ang.^3), '
                                    'a (ang.), b (ang.), c (ang.), '
                                    'alpha, beta, gamma'),
                            widths=[12,10,10,10,8,8,8],
                            file=f
                        )
                    """

                print("**Elastic constants**", file=f)
                print("", file=f)
                for st in self.__structures:
                    yamlfile = "/".join(
                        [path_data, st["st_type"], "polymlp_elastic.yaml"]
                    )
                    if os.path.exists(yamlfile):
                        elastic_data = self.__read_elastic_yaml(yamlfile)
                        text_to_csv_table(
                            elastic_data,
                            title="Elastic constants [" + st["st_type"] + "]",
                            header="1, 2, 3, 4, 5, 6",
                            widths=[10 for i in range(6)],
                            file=f,
                        )

                f.close()

    def run(self):
        self.run_summary()
        self.run_polymlps()
        self.run_predictions()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_data",
        type=str,
        default="./",
        help="Path (output of predictions)",
    )
    args = parser.parse_args()

    web = PolymlpRepositoryWebContents(path_data=args.path_data)
    web.run()
