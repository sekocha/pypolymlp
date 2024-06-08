#!/usr/bin/env python
import numpy as np
import yaml


def check_imaginary(fname, tol=-0.1, print_log=True):

    dos = np.loadtxt(fname, dtype=float)
    occ_ids = np.where(dos[:, 1] > 0)[0]
    eigs = dos[occ_ids]
    if eigs[0, 0] < tol:
        imag_ids = np.where(eigs[:, 0] < 0.001)[0]
        if print_log:
            print(fname, eigs[0, 0], "{:f}".format(sum(eigs[imag_ids, 1])))
        return True
    return False


def parse_summary_yaml(yaml_file):

    data = yaml.safe_load(open(yaml_file))
    free_energies = [[d["temperature"], d["free_energy"]] for d in data["equilibrium"]]
    return np.array(free_energies)


class SummaryEOSYaml:

    def __init__(self, yaml_file):

        self.__data = yaml.safe_load(open(yaml_file))
        self.__equilibrium_data()

        self.__fv = self.__data_dict(key="eos_data", value_key="volume_helmholtz")
        self.__fv_fit = self.__data_dict(
            key="eos_fit_data", value_key="volume_helmholtz"
        )
        self.__gp_fit = self.__data_dict(key="eos_fit_data", value_key="pressure_gibbs")

    def __data_dict(self, key="eos", value_key="volume_helmholtz"):

        data_dict = dict()
        for d in self.__data[key]:
            data_dict[d["temperature"]] = np.array(d[value_key])
        return data_dict

    def __equilibrium_data(self):

        self.__free_energy = [
            [d["temperature"], d["free_energy"]] for d in self.__data["equilibrium"]
        ]
        self.__bulk_modulus = [
            [d["temperature"], d["bulk_modulus"]] for d in self.__data["equilibrium"]
        ]
        self.__volume = [
            [d["temperature"], d["volume"]] for d in self.__data["equilibrium"]
        ]

    @property
    def equilibrium_free_energy(self):
        return self.__free_energy

    @property
    def bulk_modulus(self):
        return self.__bulk_modulus

    @property
    def equilibrium_volum(self):
        return self.__volume

    @property
    def eos_helmholtz(self):
        return self.__fv

    @property
    def eos_fit_helmholtz(self):
        return self.__fv_fit

    @property
    def eos_fit_gibbs(self):
        return self.__gp_fit


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--files", nargs="*", type=str, default=None, help="targets"
    )
    args = parser.parse_args()

    tol = -0.1
    for fname in sorted(args.files):
        check_imaginary(fname, tol=tol, print_log=True)
