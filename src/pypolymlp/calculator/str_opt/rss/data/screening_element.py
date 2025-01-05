#!/usr/bin/env python
import argparse
import os
import signal
import warnings

import numpy as np

from pypolymlp.calculator.rss.data.common import (
    estimate_n_locals,
    parse_log_summary_yaml,
    set_emin,
)

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
    parser.add_argument(
        "--e_outlier",
        type=float,
        default=None,
        help="threshold for finding outliers",
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.yaml_file)
    summary_dict, numbers = parse_log_summary_yaml(args.yaml_file, return_numbers=True)
    n_trials = numbers["number_of_trial_structures"]
    for comp, summary in summary_dict.items():
        print(" comp =", comp)
        e_all = summary["energies"]
        e_min, e_outlier = set_emin(e_all, e_outlier=args.e_outlier)
        print(" energy threshold (outlier) =", e_outlier)
        print(" energy min. =", e_min)

        e_list = [0.005, 0.01, 0.015, 0.02] + list(np.arange(0.025, 0.301, 0.025))
        for e_th in e_list:
            e_ub = e_min + e_th
            indices = np.where(e_all < e_ub)[0]
            spg_array = [summary["space_groups"][i] for i in indices]

            n_locals_estimator = estimate_n_locals(n_trials, len(indices))

            f = open(
                output_dir
                + "/log_screening_"
                + str(round(e_th, 6)).ljust(5, "0")
                + ".yaml",
                "w",
            )
            print("numbers:", file=f)
            print("  number_of_local_minimizers:", len(indices), file=f)
            print("  number_of_locals_estimator:", n_locals_estimator, file=f)
            print("  threshold_energy_(eV/atom):", round(e_th, 6), file=f)
            print("", file=f)

            print("nonequiv_structures:", file=f)
            print("- composition:", list(comp), file=f)
            print("  structures:", file=f)
            for poscar, e, spg in zip(
                summary["poscars"][indices],
                summary["energies"][indices],
                spg_array,
            ):
                print("  - id:  ", poscar, file=f)
                print("    e:   ", e, file=f)
                print("    spg: ", spg, file=f)
            f.close()
