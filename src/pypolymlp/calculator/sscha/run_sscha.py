#!/usr/bin/env python
import os
from collections import defaultdict

import numpy as np
from phono3py.file_IO import read_fc2_from_hdf5, write_fc2_to_hdf5
from phonopy import Phonopy
from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
from symfc.solvers.solver_O2 import run_solver_dense_O2

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_io import save_sscha_yaml
from pypolymlp.core.utils import ev_to_kjmol
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_st_dict,
    st_dict_to_phonopy_cell,
)

"""
const_bortzmann = 1.380649e-23 # J K^-1
const_bortzmann_ev = 8.617333262e-5 # eV K^-1
"""


class PolymlpSSCHA:

    def __init__(
        self,
        unitcell_dict,
        supercell_matrix,
        pot=None,
        params_dict=None,
        coeffs=None,
        properties=None,
    ):

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params_dict=params_dict, coeffs=coeffs)

        self.unitcell_dict = unitcell_dict
        self.supercell_matrix = supercell_matrix

        self.unitcell = st_dict_to_phonopy_cell(unitcell_dict)
        self.phonopy = Phonopy(self.unitcell, supercell_matrix)
        self.supercell = self.phonopy.supercell
        self.n_unitcells = int(round(np.linalg.det(supercell_matrix)))

        self.supercell_dict = phonopy_cell_to_st_dict(self.supercell)
        self.supercell_dict["masses"] = self.supercell.masses
        self.supercell_dict["supercell_matrix"] = supercell_matrix
        self.supercell_dict["n_unitcells"] = self.n_unitcells

        self.n_atom = len(self.supercell.masses)
        self.fc2_basis = FCBasisSetO2(self.supercell, use_mkl=False).run()

        """ for bubble diagram
        fc3_basis = FCBasisSetO3(supercell, use_mkl=False).run()
        compress_mat_fc3 = fc3_basis.compression_matrix
        compress_eigvecs_fc3 = fc3_basis.basis_set
        """

        self.ph_real = HarmonicReal(self.supercell_dict, self.prop)
        self.ph_recip = HarmonicReciprocal(self.phonopy, self.prop)

        self.fc2 = None
        self.__sscha_dict = None
        self.__log_dict = None
        self.__history_dict = None

    def __recover_fc2(self, coefs):
        compress_mat = self.fc2_basis.compression_matrix
        compress_eigvecs = self.fc2_basis.basis_set
        fc2 = compress_eigvecs @ coefs
        fc2 = (compress_mat @ fc2).reshape((self.n_atom, self.n_atom, 3, 3))
        return fc2

    def __run_solver_fc2(self):
        """Input parameter shapes are different in FCSolverO2
        and run_solver_fc2.
        FCSolverO2: (n_samples, n_atom, 3)
        run_solver_dense_O2: (n_samples, n_atom * 3)
        """

        """
        disps: (n_samples, n_atom, 3)
        forces: (n_samples, n_atom, 3)
        """
        disps = self.ph_real.displacements.transpose((0, 2, 1))
        forces = self.ph_real.forces.transpose((0, 2, 1))
        n_samples = disps.shape[0]

        fc2_coeffs = run_solver_dense_O2(
            disps.reshape((n_samples, -1)),
            forces.reshape((n_samples, -1)),
            self.fc2_basis.compression_matrix,
            self.fc2_basis.basis_set,
        )
        fc2 = self.__recover_fc2(fc2_coeffs)
        return fc2

    def __unit_kjmol(self, e):
        return ev_to_kjmol(e) / self.n_unitcells

    def __compute_sscha_properties(self, t=1000, qmesh=[10, 10, 10], first_order=True):

        self.ph_recip.force_constants = self.fc2
        self.ph_recip.compute_thermal_properties(t=t, qmesh=qmesh)

        res_dict = {
            "temperature": t,
            "harmonic_free_energy": self.ph_recip.free_energy,  # kJ/mol
            "static_potential": self.__unit_kjmol(self.ph_real.static_potential),
            "harmonic_potential": self.__unit_kjmol(
                self.ph_real.average_harmonic_potential
            ),
            "average_potential": self.__unit_kjmol(self.ph_real.average_full_potential),
            "anharmonic_free_energy": self.__unit_kjmol(
                self.ph_real.average_anharmonic_potential
            ),
        }

        if first_order:
            res_dict["anharmonic_free_energy_exact"] = 0.0
            res_dict["free_energy"] = (
                res_dict["harmonic_free_energy"] + res_dict["anharmonic_free_energy"]
            )
        else:
            pass

        return res_dict

    def __single_iter(self, t=1000, n_samples=100, qmesh=[10, 10, 10]):

        self.ph_real.force_constants = self.fc2
        self.ph_real.run(t=t, n_samples=n_samples, eliminate_outliers=True)

        self.__sscha_dict = self.__compute_sscha_properties(t=t, qmesh=qmesh)
        fc2 = self.__run_solver_fc2()
        return fc2

    def __convergence_score(self, fc2_init, fc2_update):
        norm1 = np.linalg.norm(fc2_update - fc2_init)
        norm2 = np.linalg.norm(fc2_init)
        return norm1 / norm2

    def run_frequencies(self, qmesh=[10, 10, 10]):
        self.phonopy.force_constants = self.fc2
        self.phonopy.run_mesh(qmesh)
        mesh_dict = self.phonopy.get_mesh_dict()
        return mesh_dict["frequencies"]

    def write_dos(self, qmesh=[10, 10, 10], filename="total_dos.dat"):
        self.phonopy.force_constants = self.fc2
        self.phonopy.run_total_dos()
        self.phonopy.write_total_dos(filename=filename)

    def set_initial_force_constants(self, algorithm="harmonic", filename=None):

        if algorithm == "harmonic":
            print("Initial FCs: Harmonic")
            self.fc2 = self.ph_recip.produce_harmonic_force_constants()
        elif algorithm == "const":
            print("Initial FCs: Constants")
            n_coeffs = self.fc2_basis.basis_set.shape[1]
            coeffs_fc2 = np.ones(n_coeffs) * 10
            coeffs_fc2[1::2] *= -1
            self.fc2 = self.__recover_fc2(coeffs_fc2)
        elif algorithm == "random":
            print("Initial FCs: Random")
            n_coeffs = self.fc2_basis.basis_set.shape[1]
            coeffs_fc2 = (np.random.rand(n_coeffs) - 0.5) * 20
            self.fc2 = self.__recover_fc2(coeffs_fc2)
        elif algorithm == "file":
            print("Initial FCs: File", filename)
            self.fc2 = read_fc2_from_hdf5(filename)

    def run(
        self,
        t=1000,
        n_samples=100,
        qmesh=[10, 10, 10],
        n_loop=100,
        tol=1e-2,
        mixing=0.5,
        initialize_history=True,
    ):

        if self.fc2 is None:
            self.fc2 = self.set_initial_force_constants()

        if initialize_history:
            self.__history_dict = defaultdict(list)

        n_iter, delta = 1, 100
        while n_iter <= n_loop and delta > tol:
            print("------------- Iteration :", n_iter, "-------------")
            fc2_new = self.__single_iter(t=t, n_samples=n_samples, qmesh=qmesh)
            delta = self.__convergence_score(self.fc2, fc2_new)
            self.fc2 = fc2_new * mixing + self.fc2 * (1 - mixing)

            self.__print_progress(delta)
            for key in [
                "free_energy",
                "harmonic_potential",
                "average_potential",
                "anharmonic_free_energy",
            ]:
                self.__history_dict[key].append(self.__sscha_dict[key])

            n_iter += 1

        converge = True if delta < tol else False
        self.__log_dict = {
            "converge": converge,
            "delta": delta,
            "history": self.__history_dict,
        }

    def __print_progress(self, delta):

        disp_norms = np.linalg.norm(self.ph_real.displacements, axis=1)

        print("convergence score:      ", "{:.6f}".format(delta))
        print("displacements:")
        print("  average disp. (Ang.): ", "{:.6f}".format(np.mean(disp_norms)))
        print("  max disp. (Ang.):     ", "{:.6f}".format(np.max(disp_norms)))
        print("thermodynamic_properties:")
        print(
            "  free energy (harmonic, kJ/mol)  :",
            "{:.6f}".format(self.__sscha_dict["harmonic_free_energy"]),
        )
        print(
            "  free energy (anharmonic, kJ/mol):",
            "{:.6f}".format(self.__sscha_dict["anharmonic_free_energy"]),
        )
        print(
            "  free energy (sscha, kJ/mol)     :",
            "{:.6f}".format(self.__sscha_dict["free_energy"]),
        )

    def save_results(self, args):

        log_dir = "./sscha/" + str(self.properties["temperature"]) + "/"
        os.makedirs(log_dir, exist_ok=True)
        save_sscha_yaml(self, args, filename=log_dir + "sscha_results.yaml")
        write_fc2_to_hdf5(self.force_constants, filename=log_dir + "fc2.hdf5")
        self.write_dos(filename=log_dir + "total_dos.dat")
        freq = self.run_frequencies(qmesh=args.mesh)

        print("-------- sscha runs finished --------")
        print("Temperature:      ", self.properties["temperature"])
        print("Free energy:      ", self.properties["free_energy"])
        print("Convergence:      ", self.logs["converge"])
        print("Frequency (min):  ", "{:.6f}".format(np.min(freq)))
        print("Frequency (max):  ", "{:.6f}".format(np.max(freq)))

    @property
    def properties(self):
        return self.__sscha_dict

    @property
    def logs(self):
        return self.__log_dict

    @property
    def force_constants(self):
        return self.fc2

    @force_constants.setter
    def force_constants(self, fc2):
        """(n_atom, n_atom, 3, 3)"""
        self.fc2 = fc2


def run_sscha(
    unitcell_dict,
    supercell_matrix,
    args,
    pot=None,
    params_dict=None,
    coeffs=None,
    log=True,
):

    sscha = PolymlpSSCHA(
        unitcell_dict,
        supercell_matrix,
        pot=pot,
        params_dict=params_dict,
        coeffs=coeffs,
    )

    sscha.set_initial_force_constants(algorithm=args.init, filename=args.init_file)
    freq = sscha.run_frequencies(qmesh=args.mesh)
    print("Frequency (min):  ", "{:.6f}".format(np.min(freq)))
    print("Frequency (max):  ", "{:.6f}".format(np.max(freq)))
    print("Number of FC2 basis vectors:", sscha.fc2_basis.basis_set.shape[1])

    for temp in args.temperatures:
        print("************** Temperature:", temp, "**************")
        sscha.run(
            t=temp,
            n_samples=args.n_steps,
            qmesh=args.mesh,
            n_loop=args.max_iter,
            tol=args.tol,
            mixing=args.mixing,
        )
        print("Increasing number of samples.")
        sscha.run(
            t=temp,
            n_samples=args.n_steps_final,
            qmesh=args.mesh,
            n_loop=args.max_iter,
            tol=args.tol,
            mixing=args.mixing,
            initialize_history=False,
        )
        sscha.save_results(args)


if __name__ == "__main__":

    import argparse
    import signal

    from pypolymlp.calculator.sscha.sscha_io import (
        Restart,
        n_steps_setting,
        print_parameters,
        print_structure,
        temperature_setting,
    )
    from pypolymlp.core.interface_vasp import Poscar

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--poscar",
        type=str,
        default=None,
        help="poscar file (unit cell)",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="sscha_results.yaml file for parsing " "unitcell and supercell size.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help="polymlp.lammps file",
    )
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[10, 10, 10],
        help="q-mesh for phonon calculation",
    )
    parser.add_argument(
        "-t", "--temp", type=float, default=None, help="Temperature (K)"
    )
    parser.add_argument(
        "-t_min",
        "--temp_min",
        type=float,
        default=100,
        help="Lowest temperature (K)",
    )
    parser.add_argument(
        "-t_max",
        "--temp_max",
        type=float,
        default=2000,
        help="Highest temperature (K)",
    )
    parser.add_argument(
        "-t_step",
        "--temp_step",
        type=float,
        default=100,
        help="Temperature interval (K)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Tolerance parameter for FC convergence",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        nargs=2,
        default=None,
        help="Number of steps used in " "iterations and the last iteration",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--ascending_temp",
        action="store_true",
        help="Use ascending order of temperatures",
    )
    parser.add_argument(
        "--init",
        choices=["harmonic", "const", "random", "file"],
        default="harmonic",
        help="Initial FCs",
    )
    parser.add_argument(
        "--init_file",
        default=None,
        help="Location of fc2.hdf5 for initial FCs",
    )
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter")
    args = parser.parse_args()

    if args.poscar is not None:
        unitcell_dict = Poscar(args.poscar).get_structure()
        supercell_matrix = np.diag(args.supercell)
    elif args.yaml is not None:
        res = Restart(args.yaml)
        unitcell_dict = res.unitcell
        supercell_matrix = res.supercell_matrix
        if args.pot is None:
            args.pot = res.mlp

    n_atom = len(unitcell_dict["elements"]) * np.linalg.det(supercell_matrix)
    args = temperature_setting(args)
    args = n_steps_setting(args, n_atom)

    print_parameters(supercell_matrix, args)
    print_structure(unitcell_dict)

    run_sscha(unitcell_dict, supercell_matrix, args, pot=args.pot)
