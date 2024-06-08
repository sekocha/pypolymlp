#!/usr/bin/env python
import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.utils.structure_utils import isotropic_volume_change


class PolymlpEOS:

    def __init__(
        self,
        unitcell_dict,
        pot=None,
        params_dict=None,
        coeffs=None,
        properties=None,
    ):

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

        self.__unitcell_dict = unitcell_dict

        self.__eos_data = None
        self.__eos_fit_data = None
        self.__b0 = None
        self.__e0 = None
        self.__v0 = None

    def __set_eps(self, eps_min=0.7, eps_max=2.0, eps_int=0.03, fine_grid=True):

        if fine_grid is False:
            eps_seq = np.arange(eps_min, eps_max + 0.01, eps_int)
        else:
            eps_seq = []
            if eps_min < 0.9 and eps_max > 1.1:
                eps_seq.extend(list(np.arange(eps_min, 0.9, eps_int)))
                eps_seq.extend(list(np.arange(0.9, 1.1, eps_int / 3)))
                eps_seq.extend(list(np.arange(1.1, eps_max, eps_int)))
            elif eps_min < 0.9 and eps_max < 1.1:
                eps_seq.extend(list(np.arange(eps_min, 0.9, eps_int)))
                eps_seq.extend(list(np.arange(0.9, eps_max, eps_int / 3)))
            elif eps_min > 0.9 and eps_max > 1.1:
                eps_seq.extend(list(np.arange(eps_min, 1.1, eps_int / 3)))
                eps_seq.extend(list(np.arange(1.1, eps_max, eps_int)))
            else:
                eps_seq.extend(list(np.arange(eps_min, eps_max, eps_int / 3)))

        return eps_seq

    def run_eos_fit(self, volumes, energies):

        from pymatgen.analysis.eos import EOS

        print("EOS fitting using Vinet EOS equation")
        eos = EOS(eos_name="vinet")
        eos_fit = eos.fit(volumes, energies)
        self.__b0 = eos_fit.b0_GPa
        self.__e0 = eos_fit.e0
        self.__v0 = eos_fit.v0

        v_min, v_max = min(volumes), max(volumes)
        extrapolation = (v_max - v_min) * 0.1
        v_lb = v_min - extrapolation
        v_ub = v_max + extrapolation

        volumes_eval = np.arange(v_lb, v_ub, 0.01)
        eos_fit_data = [[vol, eos_fit.func(vol)] for vol in volumes_eval]
        eos_fit_data = np.array(eos_fit_data)
        return eos_fit_data

    def run(
        self,
        eps_min=0.7,
        eps_max=2.0,
        eps_int=0.03,
        fine_grid=True,
        eos_fit=False,
    ):

        eps_list = self.__set_eps(
            eps_min=eps_min,
            eps_max=eps_max,
            eps_int=eps_int,
            fine_grid=fine_grid,
        )
        st_dicts = [
            isotropic_volume_change(self.__unitcell_dict, eps=eps) for eps in eps_list
        ]

        energies, _, _ = self.prop.eval_multiple(st_dicts)
        volumes = np.array([st["volume"] for st in st_dicts])
        print(" eps =", np.array(eps_list))
        self.__eos_data = np.array([volumes, energies]).T

        if eos_fit:
            try:
                self.__eos_fit_data = self.run_eos_fit(volumes, energies)
            except:
                print("Warning: EOS fitting failed.")

        return self

    def __write_data_2d(self, data, stream, tag="volume_helmholtz"):

        print("  " + tag + ":", file=stream)
        for d in data:
            print("  -", list(d), file=stream)
        print("", file=stream)

    def write_eos_yaml(self, write_eos_fit=True, filename="polymlp_eos.yaml"):

        f = open(filename, "w")

        if self.__b0 is not None:
            print("equilibrium:", file=f)
            print("  bulk_modulus:", float(self.__b0), file=f)
            print("  free_energy: ", self.__e0, file=f)
            print("  volume:      ", self.__v0, file=f)
            print(
                "  n_atoms:     ",
                list(self.__unitcell_dict["n_atoms"]),
                file=f,
            )
            print("", file=f)
            print("", file=f)

        print("eos_data:", file=f)
        print("", file=f)
        self.__write_data_2d(self.__eos_data, f, tag="volume_helmholtz")
        print("", file=f)

        if write_eos_fit and self.__eos_fit_data is not None:
            print("eos_fit_data:", file=f)
            print("", file=f)
            self.__write_data_2d(self.__eos_fit_data, f, tag="volume_helmholtz")
            print("", file=f)

        f.close()


if __name__ == "__main__":

    import argparse

    from pypolymlp.core.interface_vasp import Poscar

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--poscar", type=str, default=None, help="poscar file")
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.lammps",
        help="polymlp file",
    )
    args = parser.parse_args()

    unitcell = Poscar(args.poscar).get_structure()
    eos = PolymlpEOS(unitcell, pot=args.pot)
    eos.run(eos_fit=True)
    eos.write_eos_yaml()
