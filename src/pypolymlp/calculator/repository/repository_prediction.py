#!/usr/bin/env python
import glob
import os
import shutil

import yaml

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.repository.pypolymlp_calc import run_single_structure
from pypolymlp.calculator.repository.utils.target_prototypes import get_icsd_data1
from pypolymlp.calculator.repository.utils.target_structures import (  # get_structure_list_alloy2,
    get_structure_list_element1,
)
from pypolymlp.calculator.repository.utils.yaml_io_utils import write_icsd_yaml
from pypolymlp.utils.vasp_utils import write_poscar_file


class PolymlpRepositoryPrediction:

    def __init__(self, yamlfile="polymlp_summary_convex.yaml", path_vasp="./"):

        yamldata = yaml.safe_load(open(yamlfile))["polymlps"]

        self.__pot_dict = dict()
        self.__elements = None
        for potdata in yamldata:
            path_pot = potdata["path"]
            pot = sorted(glob.glob(path_pot + "/polymlp.lammps*"))
            prop = Properties(pot=pot)
            self.__pot_dict[potdata["id"]] = {
                "properties_obj": prop,
                "path": path_pot,
            }
            if self.__elements is None:
                self.__elements = prop.params_dict["elements"]

        if len(self.__elements) == 1:
            self.__target_list = get_structure_list_element1(self.__elements, path_vasp)
            self.__icsd_list = get_icsd_data1(self.__elements, path_vasp)
        else:
            raise ValueError("not available for more than binary system")

    def copy_files(self, path_output="./"):

        print("--- Copying files ---")
        for pot_id, pot_info in self.__pot_dict.items():
            print("Polymlp:", pot_id)
            path_output_single = "/".join([path_output, pot_id]) + "/"
            os.makedirs(path_output_single + "/polymlps", exist_ok=True)
            for file1 in glob.glob(pot_info["path"] + "/polymlp*"):
                shutil.copyfile(
                    file1,
                    path_output_single + "/polymlps/" + file1.split("/")[-1],
                )

            os.makedirs(path_output_single + "/energy_dist", exist_ok=True)
            file_list = glob.glob(pot_info["path"] + "/predictions/*train*")
            with open(
                path_output_single + "/energy_dist/energy-train.dat", "w"
            ) as outfile:
                outfile.write("# DFT(eV/atom), MLP(eV/atom)\n")
                for file1 in file_list:
                    f = open(file1, "r")
                    lines = f.readlines()
                    f.close()
                    for l1 in lines[1:]:
                        outfile.write(" ".join(l1.split(" ")[:-2]) + "\n")

            file_list = glob.glob(pot_info["path"] + "/predictions/*test*")
            with open(
                path_output_single + "/energy_dist/energy-test.dat", "w"
            ) as outfile:
                outfile.write("# DFT(eV/atom), MLP(eV/atom)\n")
                for file1 in file_list:
                    f = open(file1, "r")
                    lines = f.readlines()
                    f.close()
                    for l1 in lines[1:]:
                        outfile.write(" ".join(l1.split(" ")[:-2]) + "\n")

        print("--- Copying DFT files ---")
        for st, target in self.__target_list.items():
            path_st = path_output + "/vasp/" + st + "/"
            os.makedirs(path_st, exist_ok=True)
            write_poscar_file(
                target["structure"], filename=path_st + "POSCAR", header=st
            )
        return self

    def run_icsd(self, path_output="./"):

        print("--- Running ICSD prediction ---")
        for pot_id, pot_info in self.__pot_dict.items():
            print("Polymlp:", pot_id)
            st_dicts = [target["structure"] for _, target in self.__icsd_list.items()]
            energies, _, _ = pot_info["properties_obj"].eval_multiple(st_dicts)
            energies = [
                e / sum(v["structure"]["n_atoms"])
                for e, v in zip(energies, self.__icsd_list.values())
            ]

            path_output_single = "/".join([path_output, pot_id, "predictions"]) + "/"
            write_icsd_yaml(self.__icsd_list, energies, path_output=path_output_single)

        return self

    def run_properties(self, path_output="./", run_qha=False):

        print("--- Running property prediction ---")
        for pot_id, pot_info in self.__pot_dict.items():
            for st, target in self.__target_list.items():
                print("--- Polymlp:", pot_id, "(Structure:", st + ") ---")
                path_output_single = (
                    "/".join([path_output, pot_id, "predictions", st]) + "/"
                )
                run_single_structure(
                    target["structure"],
                    properties=pot_info["properties_obj"],
                    run_qha=run_qha,
                    path_output=path_output_single,
                )
        return self

    def write_summary(self, path_output="./"):

        os.makedirs(path_output + "/polymlp_summary", exist_ok=True)
        f = open(path_output + "/polymlp_summary/prediction.yaml", "w")
        print("elements:", list(self.__elements), file=f)
        print("", file=f)

        print("polymlps:", file=f)
        for pot_id, pot_info in self.__pot_dict.items():
            print("- id:   ", pot_id, file=f)
            print("  path: ", os.path.abspath(pot_info["path"]), file=f)
        print("", file=f)

        print("structures:", file=f)
        for k, v in self.__target_list.items():
            print("- st_type:         ", k, file=f)
            print("  icsd_id:         ", v["icsd_id"], file=f)
            print("  n_atom:          ", v["n_atom"], file=f)
            print("  phonon_supercell:", v["phonon_supercell"], file=f)
        print("", file=f)

        f.close()
        return self

    def run(self, path_output="./", run_qha=False):

        self.copy_files(path_output=path_output)
        self.write_summary(path_output=path_output)

        self.run_icsd(path_output=path_output)
        self.run_properties(path_output=path_output, run_qha=run_qha)

        return self


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="polymlp_summary_convex.yaml",
        help="Summary yaml file from grid search",
    )
    parser.add_argument(
        "--path_vasp",
        type=str,
        default="./",
        help="Path (vasp data for prototype structures)",
    )
    parser.add_argument(
        "--path_output",
        type=str,
        default="./",
        help="Path (output of predictions)",
    )
    args = parser.parse_args()

    pred = PolymlpRepositoryPrediction(yamlfile=args.yaml, path_vasp=args.path_vasp)
    pred.run(path_output=args.path_output, run_qha=False)
