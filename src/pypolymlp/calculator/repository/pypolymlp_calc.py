#!/usr/bin/env python
import os

import numpy as np

from pypolymlp.calculator.compute_elastic import PolymlpElastic
from pypolymlp.calculator.compute_eos import PolymlpEOS
from pypolymlp.calculator.compute_phonon import PolymlpPhonon, PolymlpPhononQHA
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym


class PypolymlpCalc:

    def __init__(self, pot=None, params_dict=None, coeffs=None, properties=None):

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)
        self.pot = pot
        self.params_dict = params_dict
        self.coeffs = coeffs

    def compute_phonon(
        self,
        unitcell_dict,
        supercell_auto=True,
        supercell_matrix=None,
        disp=0.01,
        mesh=[10, 10, 10],
        tmin=100,
        tmax=1000,
        tstep=100,
        pdos=False,
    ):

        if supercell_auto:
            pass
        elif supercell_matrix is None:
            raise ValueError("compute_phonon: supercell_matrix is needed.")

        ph = PolymlpPhonon(unitcell_dict, supercell_matrix, properties=self.prop)
        ph.produce_force_constants(displacements=disp)
        ph.compute_properties(
            mesh=mesh, t_min=tmin, t_max=tmax, t_step=tstep, pdos=pdos
        )

    def compute_phonon_qha(
        self, unitcell_dict, supercell_auto=True, supercell_matrix=None
    ):

        _ = PolymlpPhononQHA(unitcell_dict, supercell_matrix, properties=self.prop)


def run_single_structure(
    st_dict,
    pot=None,
    params_dict=None,
    coeffs=None,
    properties=None,
    run_qha=False,
    path_output="./",
):

    if properties is not None:
        prop = properties
    else:
        prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

    os.makedirs(path_output, exist_ok=True)
    print("Mode: Geometry optimization")
    minobj = MinimizeSym(st_dict, properties=prop, relax_cell=True)
    try:
        minobj.run(gtol=1e-5)
    except ValueError:
        print("Geometry optimization has failed.")
        return 0

    if minobj.success is False:
        print("Geometry optimization has failed.")
        return 0

    if minobj.energy / sum(minobj.structure["n_atoms"]) < -20:
        print("Geometry optimization has failed. (Too low cohesive energy)")
        return 0

    minobj.write_poscar(filename=path_output + "/POSCAR_eqm")
    st_dict_eq = minobj.structure

    print("Mode: EOS")
    eos = PolymlpEOS(st_dict_eq, properties=prop)
    eos.run(eps_min=0.7, eps_max=2.5, eps_int=0.02, eos_fit=False)
    eos.write_eos_yaml(write_eos_fit=False, filename=path_output + "/polymlp_eos.yaml")

    eos.run(eps_min=0.97, eps_max=1.03, eps_int=0.003, eos_fit=True)
    eos.write_eos_yaml(
        write_eos_fit=False, filename=path_output + "/polymlp_eos_fit.yaml"
    )

    print("Mode: Elastic constant")
    el = PolymlpElastic(st_dict_eq, path_output + "POSCAR_eqm", properties=prop)
    el.run()
    el.write_elastic_constants(filename=path_output + "/polymlp_elastic.yaml")

    print("Mode: Phonon")
    supercell_matrix = np.diag([4, 4, 4])
    ph = PolymlpPhonon(st_dict_eq, supercell_matrix, properties=prop)
    ph.produce_force_constants(displacements=0.01)
    ph.compute_properties(
        mesh=[10, 10, 10],
        t_min=0,
        t_max=1000,
        t_step=50,
        path_output=path_output,
    )

    if run_qha and not ph.is_imaginary():
        print("Mode: Phonon QHA")
        qha = PolymlpPhononQHA(st_dict_eq, supercell_matrix, properties=prop)
        qha.run(
            eps_min=0.8,
            eps_max=1.2,
            eps_int=0.02,
            mesh=[10, 10, 10],
            disp=0.01,
            t_min=0,
            t_max=1000,
            t_step=10,
        )
        qha.write_qha(path_output=path_output)
    elif run_qha and ph.is_imaginary():
        print("Phonon QHA is not performed " "because imaginary modes are detected.")

    return 0


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

    prop = Properties(pot=args.pot)
    unitcell = Poscar(args.poscar).get_structure()

    run_single_structure(unitcell, properties=prop, run_qha=True)
