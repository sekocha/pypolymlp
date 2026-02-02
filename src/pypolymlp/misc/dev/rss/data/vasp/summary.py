#!/usr/bin/env python
import argparse
import glob
import os
import signal
import warnings
from collections import defaultdict

import numpy as np

from pypolymlp.calculator.rss.data.common import natural_keys
from pypolymlp.calculator.rss.data.equivalency import get_equivalency
from pypolymlp.calculator.rss.data.spg import get_space_groups
from pypolymlp.core.interface_vasp import Vasprun


def find_vasprun_xml(dir_st):

    f_summary = dir_st + "vasprun.xml.polymlp"
    if not os.path.exists(f_summary):
        f_summary = dir_st + "vasprun.xml_to_mlip"
        if not os.path.exists(f_summary):
            f_summary = dir_st + "vasprun.xml"

    return f_summary


def get_summary_single(st_id, dir_res, elements=None):

    dir_st = dir_res + "/" + st_id + "/"
    f_summary = find_vasprun_xml(dir_st)
    f_poscar = dir_st + "POSCAR"

    vasp = Vasprun(f_summary)
    e = vasp.energy
    st_dict = vasp.structure

    n_atoms_sum = np.sum(st_dict["n_atoms"])

    if elements is None:
        comp = np.array(st_dict["n_atoms"]) / n_atoms_sum
    else:
        n_atoms = [
            np.count_nonzero(np.array(st_dict["elements"]) == ele) for ele in elements
        ]
        comp = n_atoms / n_atoms_sum
    comp_tuple = tuple([round(c, 15) for c in comp])

    e = e / n_atoms_sum

    single = dict()
    single["id"] = f_poscar
    single["comp"] = comp_tuple
    single["e"] = round(e, 8)
    single["n_iter"] = 0
    single["n_atoms_sum"] = n_atoms_sum

    return single


def get_summaries(st_ids, dir_res, elements=None, summary=defaultdict(list)):

    for st_id in st_ids:
        single = get_summary_single(st_id, dir_res, elements=elements)
        summary[single["comp"]].append(single)

    for comp, values in summary.items():
        summary[comp] = sorted(values, key=lambda x: x["e"])

    return summary


def write_yaml(
    summary,
    equiv_dict,
    n_trial_total,
    n_st_total,
    filename="log_summary.yaml",
):

    f = open(filename, "w")
    print("numbers:", file=f)
    print("  number_of_trial_structures:", n_trial_total, file=f)
    print("  number_of_local_minimizers:", n_st_total, file=f)
    print("", file=f)

    print("nonequiv_structures:", file=f)

    for comp, equiv_d in sorted(equiv_dict.items(), key=lambda x: x[0], reverse=True):
        print("- composition:", list(comp), file=f)
        print("  structures:", file=f)

        # n_total = sum([len(orbit) for rep, orbit in equiv_d.items()])
        for rep, orbit in equiv_d.items():
            st_attr = summary[comp][rep]
            ids_orbit = sorted(
                [summary[comp][orb]["id"] for orb in orbit],
                key=natural_keys,
            )
            print("  - id:  ", ids_orbit[0], file=f)
            print("    e:   ", round(st_attr["e"], 5), file=f)
            print("    spg: ", list(st_attr["spg"]), file=f)
        print("", file=f)
    f.close()


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="*",
        type=str,
        default=["POSCAR"],
        help="Directories containing vasp results",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--pot",
        type=str,
        default=None,
        help="polymlp.lammps file for computing " "polynomial invariants",
    )
    parser.add_argument(
        "--elements", nargs="*", type=str, default=None, help="Elements"
    )
    parser.add_argument(
        "--tol_distance",
        type=float,
        default=0.01,
        help="Tolerance for distance in duplicate " "structure elimination",
    )
    parser.add_argument(
        "--tol_energy",
        type=float,
        default=0.001,
        help="Tolerance for energy in duplicate " "structure elimination",
    )
    args = parser.parse_args()

    summary = defaultdict(list)
    for dir1 in args.dirs:
        targets = sorted(glob.glob(dir1 + "/*"), key=natural_keys)
        st_ids = [t.split("/")[-1] for t in targets]
        summary = get_summaries(st_ids, dir1, elements=args.elements, summary=summary)

    n_trial_total = sum([len(values) for values in summary.values()])

    for comp, values in summary.items():
        poscars = [v["id"] for v in values]
        spgs = get_space_groups(poscars)
        for v, s in zip(values, spgs):
            v["spg"] = s

    n_st_total = 0
    equiv_dict = dict()
    for comp, values in summary.items():
        print(" comp =", comp)
        equiv_dict[comp] = get_equivalency(
            values,
            pot=args.pot,
            tol_distance=args.tol_distance,
            tol_energy=args.tol_energy,
            pmg_matcher=False,
            verbose=True,
        )
        n_st_total += len(equiv_dict[comp])

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = "./"

    output_file = output_dir + "/log_summary.yaml"
    print(" directory for output files:", output_dir)

    write_yaml(summary, equiv_dict, n_trial_total, n_st_total, filename=output_file)
