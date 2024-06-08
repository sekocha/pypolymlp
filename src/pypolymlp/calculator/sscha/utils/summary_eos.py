#!/usr/bin/env python
import argparse
import glob
from collections import defaultdict

import numpy as np
from pymatgen.analysis.eos import EOS

from pypolymlp.calculator.sscha.utils.summary import get_restart_objects
from pypolymlp.calculator.sscha.utils.utils import check_imaginary

evang3_to_GPa = 160.2176634


def press(eos_fit, v0, eps=1e-4):
    vm, vp = v0 - eps, v0 + eps
    deriv = (eos_fit.func(vp) - eos_fit.func(vm)) / (2 * eps)
    return -deriv


def fit_eos(data, n_min=5):
    if len(data) < n_min:
        return None

    eos = EOS(eos_name="vinet")
    try:
        eos_fit = eos.fit(data[:, 0], data[:, 1])
    except:
        eos_fit = None
    return eos_fit


def write_data_2d(data, stream, temp=None, tag="volume_free_energy"):
    if temp is not None:
        print("- temperature:", temp, file=stream)
    print("  " + tag + ":", file=stream)
    for d in data:
        print("  -", list(d), file=stream)
    print("", file=stream)


def write_eos_yaml(eos_fit_array, filename="volume_dependence.yaml"):

    f = open(filename, "w")

    print("equilibrium:", file=f)
    for temp, data, eos_fit in eos_fit_array:
        print("", file=f)
        print("- temperature:", temp, file=f)
        print("  bulk_modulus:", float(eos_fit.b0_GPa), file=f)
        print("  free_energy: ", eos_fit.e0, file=f)
        print("  volume:      ", eos_fit.v0, file=f)
    print("", file=f)
    print("", file=f)

    print("eos_data:", file=f)
    print("", file=f)
    for temp, data, _ in eos_fit_array:
        write_data_2d(data, f, temp=temp, tag="volume_helmholtz")

    print("", file=f)

    print("eos_fit_data:", file=f)
    print("", file=f)
    for temp, data, eos_fit in eos_fit_array:
        data = np.array(data)
        minv, maxv = min(data[:, 0]), max(data[:, 0])
        extra = (maxv - minv) * 0.3
        ev_fit, gp_fit = [], []
        for vol in np.arange(minv - extra, maxv + extra, 0.01):
            helm = eos_fit.func(vol)
            pressure = press(eos_fit, vol)
            ev_fit.append([vol, helm])
            gp_fit.append([pressure * evang3_to_GPa, helm + pressure * vol])
        write_data_2d(ev_fit, f, temp=temp, tag="volume_helmholtz")
        write_data_2d(gp_fit, f, tag="pressure_gibbs")
    print("", file=f)

    f.close()


def write_yaml(ftotal_all, filename="volume_dependence.yaml"):

    eos_fit_array = []
    for temp, data in ftotal_all.items():
        print("EOS fit: Temperature:", temp)
        eos_fit = fit_eos(np.array(data))
        if eos_fit is not None:
            eos_fit_array.append([temp, data, eos_fit])

    temps = np.array([t for t, _, _ in eos_fit_array])
    sortid = temps.argsort()
    eos_fit_array = [eos_fit_array[i] for i in sortid]
    write_eos_yaml(eos_fit_array, filename=filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="*",
        type=str,
        default=None,
        help="Directories containing sscha results",
    )
    """
    electronic_free_energy.dat
    format: [[temp1, f_el1], [temp2, f_el2], ...]
    """
    parser.add_argument(
        "--electronic",
        type=str,
        default=None,
        help="Result file name for electronic free energy" " (unit: eV/cell)",
    )
    parser.add_argument(
        "--electronic_unit",
        type=str,
        default="eV/cell",
        help="Electronic free energy unit",
    )

    args = parser.parse_args()

    if args.dirs is None:
        dirs = sorted(glob.glob("./*/"))
    else:
        dirs = [d + "/" for d in args.dirs]
    print("Directories:", dirs)

    """ Parsing sscha_results.yaml files """
    res_volumes = []
    for d in dirs:
        yml_files = sorted(glob.glob(d + "/sscha/*/sscha_results.yaml"))
        res_volumes.append(get_restart_objects(yml_files, unit="eV/atom"))

    """ Setting electronic free energies """
    if args.electronic is None:
        elf_volumes = [None for d in dirs]
    else:
        elf_volumes = [dict(np.loadtxt(d + args.electronic)) for d in dirs]
        if args.electronic_unit == "eV/cell":
            for elf, res in zip(elf_volumes, res_volumes):
                n_atom_unitcell = sum(res[0].unitcell["n_atoms"])
                for k in elf.keys():
                    elf[k] /= n_atom_unitcell
        else:
            raise ValueError("--electronic_unit eV/cell is available.")

    ftotal_all = defaultdict(list)
    ftotal_all_noimag = defaultdict(list)
    if args.electronic is not None:
        ftotal_all_with_elf = defaultdict(list)
        ftotal_all_with_elf_noimag = defaultdict(list)

    for d, res_array, elf in zip(dirs, res_volumes, elf_volumes):
        dosfiles = sorted(glob.glob(d + "/sscha/*/total_dos.dat"))
        imaginaries = [check_imaginary(dosfile) for dosfile in dosfiles]

        for res, imag in zip(res_array, imaginaries):
            temp = res.temperature
            f_sum = res.free_energy + res.static_potential
            volume = res.unitcell_volume

            ftotal_all[temp].append([volume, f_sum])
            if imag is False:
                ftotal_all_noimag[temp].append([volume, f_sum])

        if args.electronic is not None:
            for res, imag in zip(res_array, imaginaries):
                temp = res.temperature
                f_sum = res.free_energy + res.static_potential + elf[temp]
                volume = res.unitcell_volume

                ftotal_all_with_elf[temp].append([volume, f_sum])
                if imag is False:
                    ftotal_all_with_elf_noimag[temp].append([volume, f_sum])

    write_yaml(ftotal_all, filename="summary_eos.yaml")
    write_yaml(ftotal_all_noimag, filename="summary_eos_noimag.yaml")
    if args.electronic is not None:
        write_yaml(ftotal_all_with_elf, filename="summary_eos_elf.yaml")
        write_yaml(
            ftotal_all_with_elf_noimag,
            filename="summary_eos_noimag_elf.yaml",
        )
