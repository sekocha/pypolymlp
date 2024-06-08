#!/usr/bin/env python
import os
from collections import defaultdict
from math import acos, degrees

import numpy as np
import spglib
import yaml

from pypolymlp.calculator.repository.utils.figure_utils_each_mlp import (
    plot_energy,
    plot_eos,
    plot_eos_separate,
    plot_icsd_prediction,
    plot_phonon,
    plot_phonon_qha_bulk_modulus,
    plot_phonon_qha_thermal_expansion,
)
from pypolymlp.calculator.repository.utils.figure_utils_summary import (
    plot_eqm_properties,
    plot_mlp_distribution,
)
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import st_dict_to_phonopy_cell
from pypolymlp.utils.structure_utils import get_lattice_constants


class PolymlpRepositoryGeneration:

    def __init__(self, path_data="./"):

        self.__path_data = path_data
        yamlfile = path_data + "/polymlp_summary_convex.yaml"
        yamldata_convex = self.__read_yaml(yamlfile)
        self.__system = yamldata_convex["system"]
        self.__yamldata_convex = yamldata_convex["polymlps"]

        self.__pot_ids = [d["id"] for d in self.__yamldata_convex]
        self.__costs = [float(d["cost_single"]) for d in self.__yamldata_convex]

        yamlfile = path_data + "/polymlp_summary/prediction.yaml"
        if os.path.exists(yamlfile):
            yamldata = self.__read_yaml(yamlfile)
            self.__target_list = yamldata["structures"]
        else:
            print("Warning: structure list is not found.")

    def __read_yaml(self, yamlfile):

        f = open(yamlfile)
        yamldata = yaml.safe_load(f)
        f.close()
        return yamldata

    def __single_lattice_constants(
        self, poscar, filename="polymlp_lattice_constants.yaml"
    ):

        try:
            cell = Poscar(poscar).get_structure()
        except:
            return 0

        cell_ph = st_dict_to_phonopy_cell(cell)

        lattice, _, _ = spglib.standardize_cell(
            (cell_ph.cell, cell_ph.scaled_positions, cell_ph.numbers),
            to_primitive=False,
        )
        cell["axis"] = lattice.T
        a, b, c, calpha, cbeta, cgamma = get_lattice_constants(cell)
        volume = np.linalg.det(cell["axis"])

        self.__write_lattice_constants_yaml(
            a, b, c, calpha, cbeta, cgamma, volume, filename=filename
        )
        return 1

    def __write_lattice_constants_yaml(
        self,
        a,
        b,
        c,
        calpha,
        cbeta,
        cgamma,
        volume,
        filename="polymlp_lattice_constants.yaml",
    ):
        f = open(filename, "w")
        print("standardized_lattice_constants:", file=f)
        print("  a:", a, file=f)
        print("  b:", b, file=f)
        print("  c:", c, file=f)
        print("  alpha:", degrees(acos(calpha)), file=f)
        print("  beta:", degrees(acos(cbeta)), file=f)
        print("  gamma:", degrees(acos(cgamma)), file=f)
        print("  volume:", volume, file=f)

        f.close()

    def run_lattice_constants(self):

        for target in self.__target_list:
            st = target["st_type"]
            for pot_id, cost in zip(self.__pot_ids, self.__costs):
                path_output = (
                    "/".join([self.__path_data, pot_id, "predictions", st]) + "/"
                )
                POSCAR_file = path_output + "/POSCAR_eqm"
                self.__single_lattice_constants(
                    POSCAR_file,
                    filename=path_output + "polymlp_lattice_constants.yaml",
                )

            path_output = "/".join([self.__path_data, "vasp", st]) + "/"
            POSCAR_file = path_output + "/POSCAR"
            self.__single_lattice_constants(
                POSCAR_file,
                filename=path_output + "polymlp_lattice_constants.yaml",
            )

    def run_mlp_distribution(self, dpi=300):

        yamlfile = self.__path_data + "/polymlp_summary_all.yaml"
        yamldata = self.__read_yaml(yamlfile)["polymlps"]

        d_all = [
            [
                d["cost_single"],
                d["cost_openmp"],
                d["rmse_energy"],
                d["rmse_force"],
                d["id"],
            ]
            for d in yamldata
        ]
        d_convex = [
            [
                d["cost_single"],
                d["cost_openmp"],
                d["rmse_energy"],
                d["rmse_force"],
                d["id"],
            ]
            for d in self.__yamldata_convex
        ]
        d_all, d_convex = np.array(d_all), np.array(d_convex)

        path_output = self.__path_data + "/polymlp_summary/"
        plot_mlp_distribution(
            d_all,
            d_convex,
            self.__system,
            path_output=path_output,
            dpi=dpi,
        )

        return self

    def run_eos(self, dpi=300):

        eos_dict = defaultdict(dict)
        e_eqm_dict = defaultdict(dict)
        eqm_props_dict = dict()
        for target in self.__target_list:
            st = target["st_type"]
            eqm_props = []
            for pot_id, cost in zip(self.__pot_ids, self.__costs):
                yamlfile = "/".join(
                    [
                        self.__path_data,
                        pot_id,
                        "predictions",
                        st,
                        "polymlp_eos_fit.yaml",
                    ]
                )
                if os.path.exists(yamlfile):
                    yamldata = self.__read_yaml(yamlfile)

                    if "equilibrium" in yamldata:
                        eqm_data = yamldata["equilibrium"]
                        n_atom_sum = sum([int(n) for n in eqm_data["n_atoms"]])
                        energy = float(eqm_data["free_energy"]) / n_atom_sum
                        volume = float(eqm_data["volume"]) / n_atom_sum
                        bm = float(eqm_data["bulk_modulus"])
                        eqm_props.append([cost, energy, volume, bm])
                        e_eqm_dict[pot_id][st] = energy

                    yamlfile = "/".join(
                        [
                            self.__path_data,
                            pot_id,
                            "predictions",
                            st,
                            "polymlp_eos.yaml",
                        ]
                    )
                    yamldata = self.__read_yaml(yamlfile)
                    eos_data = yamldata["eos_data"]["volume_helmholtz"]
                    eos_data = np.array(eos_data, dtype=float) / n_atom_sum
                    eos_dict[pot_id][st] = eos_data

            eqm_props_dict[st] = np.array(eqm_props)

        path_output = self.__path_data + "/polymlp_summary/"
        plot_eqm_properties(
            eqm_props_dict,
            self.__system,
            path_output=path_output,
            dpi=dpi,
        )

        # emin = ([np.min(prop[:,1]) for prop in eqm_props_dict.values()])
        for pot_id in self.__pot_ids:
            path_output = "/".join([self.__path_data, pot_id, "predictions"])
            if pot_id in eos_dict:
                emin = min(e_eqm_dict[pot_id].values())
                plot_eos(
                    eos_dict[pot_id],
                    self.__system,
                    pot_id,
                    emin=emin,
                    path_output=path_output,
                    dpi=dpi,
                )
                plot_eos_separate(
                    eos_dict[pot_id],
                    self.__system,
                    pot_id,
                    emin=emin,
                    path_output=path_output,
                    dpi=dpi,
                )

        return self

    def run_energy_distribution(self, dpi=300):

        for pot_id in self.__pot_ids:
            path_output = "/".join([self.__path_data, pot_id, "energy_dist"])
            file_train = "/".join([path_output, "energy-train.dat"])
            file_test = "/".join([path_output, "energy-test.dat"])
            data_train = np.loadtxt(file_train, dtype=float, skiprows=1)
            data_test = np.loadtxt(file_test, dtype=float, skiprows=1)

            plot_energy(
                data_train,
                data_test,
                self.__system,
                pot_id,
                path_output=path_output,
                dpi=dpi,
            )
        return self

    def run_icsd_prediction(self, dpi=300):

        for pot_id in self.__pot_ids:
            path_output = "/".join([self.__path_data, pot_id, "predictions"])
            yamlfile = "/".join([path_output, "polymlp_icsd_pred.yaml"])
            if os.path.exists(yamlfile):
                yamldata = self.__read_yaml(yamlfile)
                icsd_dict = yamldata["icsd_predictions"]
                plot_icsd_prediction(
                    icsd_dict,
                    self.__system,
                    pot_id,
                    path_output=path_output,
                    dpi=dpi,
                    figsize=(10, 4),
                )

        return self

    def run_phonon(self, dpi=300):

        for pot_id in self.__pot_ids:
            phonon_dict = dict()
            path_output = "/".join([self.__path_data, pot_id, "predictions"])
            for target in self.__target_list:
                st = target["st_type"]
                yamlfile = "/".join(
                    [
                        path_output,
                        st,
                        "polymlp_phonon/thermal_properties.yaml",
                    ]
                )
                if os.path.exists(yamlfile):
                    yamldata = self.__read_yaml(yamlfile)

                    n_atom = int(yamldata["natom"])
                    datafile = "/".join(
                        [path_output, st, "polymlp_phonon/total_dos.dat"]
                    )
                    phonon_dict[st] = np.loadtxt(datafile, dtype=float, skiprows=1)
                    phonon_dict[st][:, 1] /= n_atom

            plot_phonon(
                phonon_dict,
                self.__system,
                pot_id,
                path_output=path_output,
                dpi=dpi,
            )

        return self

    def run_phonon_qha(self, dpi=300):

        for pot_id in self.__pot_ids:
            thermal_expansion_dict = dict()
            bm_dict = dict()
            path_output = "/".join([self.__path_data, pot_id, "predictions"])
            for target in self.__target_list:
                st = target["st_type"]
                datafile = "/".join(
                    [
                        path_output,
                        st,
                        "polymlp_phonon_qha/thermal_expansion.dat",
                    ]
                )
                if os.path.exists(datafile):
                    thermal_expansion_dict[st] = np.loadtxt(datafile, dtype=float)
                datafile = "/".join(
                    [
                        path_output,
                        st,
                        "polymlp_phonon_qha/bulk_modulus-temperature.dat",
                    ]
                )
                if os.path.exists(datafile):
                    bm_dict[st] = np.loadtxt(datafile, dtype=float)

            if len(bm_dict) > 0:
                plot_phonon_qha_thermal_expansion(
                    thermal_expansion_dict,
                    self.__system,
                    pot_id,
                    path_output=path_output,
                    dpi=dpi,
                )
                plot_phonon_qha_bulk_modulus(
                    bm_dict,
                    self.__system,
                    pot_id,
                    path_output=path_output,
                    dpi=dpi,
                )

        return self

    def run(self):

        self.run_mlp_distribution()
        self.run_lattice_constants()
        self.run_eos()
        self.run_energy_distribution()
        self.run_icsd_prediction()
        self.run_phonon()
        self.run_phonon_qha()


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

    pred = PolymlpRepositoryGeneration(path_data=args.path_data)
    pred.run()
    # pred.run_lattice_constants()
    # pred.run_phonon_qha()
