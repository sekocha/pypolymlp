#!/usr/bin/env python
import os


def write_icsd_yaml(icsd_list, energies_mlp, path_output="./"):

    os.makedirs(path_output, exist_ok=True)
    f = open(path_output + "/polymlp_icsd_pred.yaml", "w")
    print("unit: eV/atom", file=f)
    print("", file=f)
    print("icsd_predictions:", file=f)
    for e, (st_key, val) in zip(energies_mlp, icsd_list.items()):
        print("- prototype:", st_key, file=f)
        print("  dft:", val["DFT_energy"], file=f)
        print("  mlp:", e, file=f)
        print("", file=f)
    f.close()
