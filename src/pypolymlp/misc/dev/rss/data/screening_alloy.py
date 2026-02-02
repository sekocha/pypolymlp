#!/usr/bin/env python
import argparse
import copy
import os
import signal
import warnings

import numpy as np

from pypolymlp.calculator.rss.data.alloy import AlloyEnergy
from pypolymlp.calculator.rss.data.common import (  # estimate_n_locals,
    parse_log_summary_yaml,
    set_emin,
)


def write_hull_yaml(data_ch, e_form_ch, e_info, filename="log_hull.yaml"):

    data = sorted(
        [[tuple(d[:-1]), d[-1], ef] for d, ef in zip(data_ch, e_form_ch)],
        reverse=True,
    )

    f = open(filename, "w")
    print("nonequiv_structures:", file=f)
    for comp, e, ef in data:
        print("- composition:", list(comp), file=f)
        print("  structures:", file=f)
        print("  - id:         ", e_info[comp]["id_min"], file=f)
        print("    e:          ", "{:.5f}".format(e), file=f)
        print("    e_formation:", "{:.5f}".format(ef), file=f)
        print("", file=f)
    f.close()


def write_summary_yaml(summary_dict, numbers_dict=None, filename="log_screening.yaml"):

    f = open(filename, "w")
    if numbers_dict is not None:
        print("numbers:", file=f)
        print(
            "  number_of_trial_structures:",
            numbers_dict["number_of_trial_structures"],
            file=f,
        )
        print(
            "  number_of_local_minimizers:",
            numbers_dict["number_of_local_minimizers"],
            file=f,
        )
        print(
            "  threshold                 :",
            numbers_dict["threshold"],
            file=f,
        )
        print("", file=f)

    print("nonequiv_structures:", file=f)
    for comp, summary in sorted(summary_dict.items(), key=lambda x: x[0], reverse=True):
        if len(summary["poscars"]) > 0:
            print("- composition:", list(comp), file=f)
            print("  structures:", file=f)
            for id1, e1, spg1, ef1, eah1 in zip(
                summary["poscars"],
                summary["energies"],
                summary["space_groups"],
                summary["energies_formation"],
                summary["energies_abovehull"],
            ):
                print("  - id:         ", id1, file=f)
                print("    e:          ", "{:.5f}".format(e1), file=f)
                print("    spg:        ", spg1, file=f)
                print("    e_formation:", "{:.5f}".format(ef1), file=f)
                print("    e_abovehull:", "{:.5f}".format(eah1), file=f)
            print("", file=f)
    f.close()


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--yaml_file",
        type=str,
        default="log_summary.yaml",
        help="summary yaml file",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.yaml_file)
    summary_dict, numbers = parse_log_summary_yaml(args.yaml_file, return_numbers=True)
    n_trials = numbers["number_of_trial_structures"]

    e_info = dict()
    data_min = []
    """Finding minimum structure for each composition"""
    for comp, summary in summary_dict.items():
        e_all = summary["energies"]
        e_min, e_outlier = set_emin(e_all)
        match = e_all > e_outlier
        id_min = np.argmin(e_all[match])

        e_info[comp] = dict()
        e_info[comp]["min"] = e_min
        e_info[comp]["outlier"] = e_outlier
        e_info[comp]["id_min"] = summary["poscars"][id_min]

        if e_outlier > -np.inf:
            print("Composition =", comp)
            print(" Energy min. =", e_min)
            print(" Energy threshold (outlier) =", e_outlier)
            print(" Energy (< e_outlier) =", e_all[e_all < e_outlier])

        add_entry = list(comp)
        add_entry.append(e_min)
        data_min.append(add_entry)
    data_min = np.array(data_min).astype(float)

    """Evaluating convex hull"""
    alloy = AlloyEnergy()
    data_ch, _ = alloy.compute_ch(data_min=data_min)
    e_form_ch = alloy.compute_formation_e(data_ch)
    write_hull_yaml(data_ch, e_form_ch, e_info)

    """Evaluating E_above_convex_hull"""
    alloy.initialize_composition_partition()
    for comp_tuple, summary in summary_dict.items():
        e_all = summary["energies"]
        comp = np.array(comp_tuple).astype(float)
        summary["energies_formation"] = alloy.compute_formation_e2(comp, e_all)
        e_hull = alloy.get_convex_hull_energy(comp)
        summary["energies_abovehull"] = e_all - e_hull

    write_summary_yaml(summary_dict, filename="log_screening_inf.yaml")

    e_list = [0.005, 0.01, 0.015, 0.02] + list(np.arange(0.025, 0.301, 0.025))
    for e_th in e_list:
        summary_dict_th = copy.deepcopy(summary_dict)
        n_locals = 0
        for comp_tuple, summary in summary_dict_th.items():
            e_abovehull = summary["energies_abovehull"]
            ids = np.where(e_abovehull < e_th)[0]
            summary["energies_abovehull"] = e_abovehull[ids]
            summary["energies"] = summary["energies"][ids]
            summary["energies_formation"] = summary["energies_formation"][ids]
            summary["space_groups"] = [summary["space_groups"][i] for i in ids]
            summary["poscars"] = summary["poscars"][ids]
            n_locals += len(summary["energies_abovehull"])

        numbers_dict = dict()
        numbers_dict["number_of_trial_structures"] = n_trials
        numbers_dict["number_of_local_minimizers"] = n_locals
        numbers_dict["threshold"] = round(e_th, 6)

        filename = "log_screening_" + str(round(e_th, 6)).ljust(5, "0") + ".yaml"
        write_summary_yaml(
            summary_dict_th, numbers_dict=numbers_dict, filename=filename
        )
