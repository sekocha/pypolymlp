"""A series of property calculations for a single structure."""

import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.compute_elastic import PolymlpElastic
from pypolymlp.calculator.compute_eos import PolymlpEOS
from pypolymlp.calculator.compute_phonon import PolymlpPhonon, PolymlpPhononQHA
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure


def run_single_structure(
    structure: PolymlpStructure,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    run_qha: bool = False,
    path_output: str = "./",
    verbose: bool = True,
):
    """A series of property calculations for a single structure.

    Parameters
    ----------
    structure: Structure in PolymlpStructure format
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties object.

    Any one of pot, (params, coeffs), and properties is required.
    """

    if properties is not None:
        prop = properties
    else:
        prop = Properties(pot=pot, params=params, coeffs=coeffs)

    os.makedirs(path_output, exist_ok=True)
    if verbose:
        print("Mode: Geometry optimization")

    minobj = MinimizeSym(structure, properties=prop, relax_cell=True)
    try:
        minobj.run(gtol=1e-5)
    except ValueError:
        print("Geometry optimization has failed.")
        return 0

    if minobj.success == False:
        print("Geometry optimization has failed.")
        return 0

    if minobj.energy / sum(minobj.structure.n_atoms) < -20:
        print("Geometry optimization has failed. (Too low cohesive energy)")
        return 0

    minobj.write_poscar(filename=path_output + "/POSCAR_eqm")
    structure_eq = minobj.structure

    if verbose:
        print("Mode: EOS")
    eos = PolymlpEOS(structure_eq, properties=prop)
    eos.run(eps_min=0.7, eps_max=2.5, eps_int=0.02, eos_fit=False)
    eos.write_eos_yaml(write_eos_fit=False, filename=path_output + "/polymlp_eos.yaml")

    eos.run(eps_min=0.97, eps_max=1.03, eps_int=0.003, eos_fit=True)
    eos.write_eos_yaml(
        write_eos_fit=False, filename=path_output + "/polymlp_eos_fit.yaml"
    )

    if verbose:
        print("Mode: Elastic constant")
    el = PolymlpElastic(structure_eq, path_output + "POSCAR_eqm", properties=prop)
    el.run()
    el.write_elastic_constants(filename=path_output + "/polymlp_elastic.yaml")

    if verbose:
        print("Mode: Phonon")
    supercell_matrix = np.diag([4, 4, 4])
    ph = PolymlpPhonon(structure_eq, supercell_matrix, properties=prop)
    ph.produce_force_constants(displacements=0.01)
    ph.compute_properties(
        mesh=[10, 10, 10],
        t_min=0,
        t_max=1000,
        t_step=50,
        path_output=path_output,
    )

    if run_qha and not ph.is_imaginary():
        if verbose:
            print("Mode: Phonon QHA")
        qha = PolymlpPhononQHA(structure_eq, supercell_matrix, properties=prop)
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
        if verbose:
            print("Phonon QHA is not performed because imaginary modes are detected.")

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
    unitcell = Poscar(args.poscar).structure

    run_single_structure(unitcell, properties=prop, run_qha=True)
