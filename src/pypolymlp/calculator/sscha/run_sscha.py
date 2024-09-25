"""Class for performing SSCHA."""

import os
from typing import Optional

import numpy as np
from phono3py.file_IO import read_fc2_from_hdf5, write_fc2_to_hdf5
from phonopy import Phonopy
from symfc import Symfc

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_utils import PolymlpDataSSCHA, save_sscha_yaml
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.utils import ev_to_kjmol
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)


class PolymlpSSCHA:

    def __init__(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: Unitcell in PolymlpStructure format
        supercell_matrix: Supercell matrix.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        self._verbose = verbose
        self._unitcell = unitcell
        self._supercell_matrix = supercell_matrix
        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self.phonopy = Phonopy(structure_to_phonopy_cell(unitcell), supercell_matrix)
        self.n_unitcells = int(round(np.linalg.det(supercell_matrix)))
        self._n_atom = len(self.phonopy.supercell.masses)

        self.supercell_polymlp = phonopy_cell_to_structure(self.phonopy.supercell)
        self.supercell_polymlp.masses = self.phonopy.supercell.masses
        self.supercell_polymlp.supercell_matrix = supercell_matrix
        self.supercell_polymlp.n_unitcells = self.n_unitcells

        self._symfc = Symfc(self.phonopy.supercell, use_mkl=True, log_level=0)
        self._symfc.compute_basis_set(2)

        self.ph_real = HarmonicReal(self.supercell_polymlp, self.prop)
        self.ph_recip = HarmonicReciprocal(self.phonopy, self.prop)

        self.fc2 = None
        self._sscha_current = None
        self._sscha_log = []

    def _unit_kjmol(self, e):
        return ev_to_kjmol(e) / self.n_unitcells

    def _compute_sscha_properties(self, t=1000, qmesh=[10, 10, 10]):

        self.ph_recip.force_constants = self.fc2
        self.ph_recip.compute_thermal_properties(t=t, qmesh=qmesh)

        res = PolymlpDataSSCHA(
            temperature=t,
            static_potential=self._unit_kjmol(self.ph_real.static_potential),
            harmonic_potential=self._unit_kjmol(
                self.ph_real.average_harmonic_potential
            ),
            harmonic_free_energy=self.ph_recip.free_energy,  # kJ/mol
            average_potential=self._unit_kjmol(self.ph_real.average_full_potential),
            anharmonic_free_energy=self._unit_kjmol(
                self.ph_real.average_anharmonic_potential
            ),
        )
        return res

    def _run_solver_fc2(self):
        """Estimate FC2 from a forces-displacements dataset."""
        self._symfc.displacements = self.ph_real.displacements.transpose((0, 2, 1))
        self._symfc.forces = self.ph_real.forces.transpose((0, 2, 1))
        self._symfc.solve(2, is_compact_fc=False)
        return self._symfc.force_constants[2]

    def _recover_fc2(self, coefs):
        """Recover FC2 from coefficients for basis set."""
        basis_set = self._symfc.basis_set[2]
        compr_mat = basis_set.compression_matrix
        fc = basis_set.basis_set @ coefs
        return (compr_mat @ fc).reshape((self._n_atom, self._n_atom, 3, 3))

    def _convergence_score(self, fc2_init, fc2_update):
        norm1 = np.linalg.norm(fc2_update - fc2_init)
        norm2 = np.linalg.norm(fc2_init)
        return norm1 / norm2

    def _single_iter(
        self,
        t=1000,
        n_samples=100,
        qmesh=[10, 10, 10],
        mixing=0.5,
    ) -> tuple[np.ndarray, float]:
        """Run a standard single sscha iteration."""

        self.ph_real.force_constants = self.fc2
        self.ph_real.run(t=t, n_samples=n_samples, eliminate_outliers=True)
        self._sscha_current = self._compute_sscha_properties(t=t, qmesh=qmesh)

        if self._verbose:
            print("Running symfc solver.")
        fc2_new = self._run_solver_fc2()
        fc2_new = fc2_new * mixing + self.fc2 * (1 - mixing)
        delta = self._convergence_score(self.fc2, fc2_new)
        self._sscha_current.delta = delta

        self.fc2 = fc2_new
        self._sscha_log.append(self._sscha_current)
        return self.fc2

    def _print_progress(self):

        disp_norms = np.linalg.norm(self.ph_real.displacements, axis=1)

        print("temperature:      ", "{:.1f}".format(self._sscha_current.temperature))
        print("number of samples:", disp_norms.shape[0])
        print("convergence score:      ", "{:.6f}".format(self._sscha_current.delta))
        print("displacements:")
        print("  average disp. (Ang.): ", "{:.6f}".format(np.mean(disp_norms)))
        print("  max disp. (Ang.):     ", "{:.6f}".format(np.max(disp_norms)))
        print("thermodynamic_properties:")
        print(
            "  free energy (harmonic, kJ/mol)  :",
            "{:.6f}".format(self._sscha_current.harmonic_free_energy),
        )
        print(
            "  free energy (anharmonic, kJ/mol):",
            "{:.6f}".format(self._sscha_current.anharmonic_free_energy),
        )
        print(
            "  free energy (sscha, kJ/mol)     :",
            "{:.6f}".format(self._sscha_current.free_energy),
        )

    def set_initial_force_constants(self, algorithm="harmonic", filename=None):
        """Set initial FC2."""

        if algorithm == "harmonic":
            if self._verbose:
                print("Initial FCs: Harmonic")
            self.fc2 = self.ph_recip.produce_harmonic_force_constants()
        elif algorithm == "const":
            if self._verbose:
                print("Initial FCs: Constants")
            n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
            coeffs_fc2 = np.ones(n_coeffs) * 10
            coeffs_fc2[1::2] *= -1
            self.fc2 = self._recover_fc2(coeffs_fc2)
        elif algorithm == "random":
            if self._verbose:
                print("Initial FCs: Random")
            n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
            coeffs_fc2 = (np.random.rand(n_coeffs) - 0.5) * 20
            self.fc2 = self._recover_fc2(coeffs_fc2)
        elif algorithm == "file":
            if self._verbose:
                print("Initial FCs: File", filename)
            self.fc2 = read_fc2_from_hdf5(filename)

    def run(
        self,
        t: int = 1000,
        n_samples: int = 100,
        qmesh: tuple[int, int, int] = (10, 10, 10),
        max_iter: int = 100,
        tol: float = 1e-2,
        mixing: float = 0.5,
        initialize_history: bool = True,
    ):
        """Run sscha iterations.

        Parameters
        ----------
        t: Temperature (K).
        n_samples: Number of sample supercells.
        qmesh: q-point mesh to calculate harmonic properties from effective FC2s.
        max_iter: Maximum number of iterations in SSCHA.
        tol: Convergence tolerance for FC2.
        mixing: Mixing parameter.
                FCs are updated by FC2 = FC2(new) * mixing + FC2(old) * (1-mixing).
        initialize_history: Initialize history logs.
        """

        if self.fc2 is None:
            self.fc2 = self.set_initial_force_constants()

        if initialize_history:
            self._sscha_log = []

        n_iter, delta = 1, 1e10
        while n_iter <= max_iter and delta > tol:
            if self._verbose:
                print("------------- Iteration :", n_iter, "-------------")

            self.fc2 = self._single_iter(
                t=t,
                n_samples=n_samples,
                qmesh=qmesh,
                mixing=mixing,
            )
            if self._verbose:
                self._print_progress()

            delta = self._sscha_current.delta
            n_iter += 1

        converge = True if delta < tol else False
        self._sscha_log[-1].converge = converge

    def _run_frequencies(self, qmesh=[10, 10, 10]):
        self.phonopy.force_constants = self.fc2
        self.phonopy.run_mesh(qmesh)
        mesh_dict = self.phonopy.get_mesh_dict()
        return mesh_dict["frequencies"]

    def _write_dos(self, qmesh=[10, 10, 10], filename="total_dos.dat"):
        self.phonopy.force_constants = self.fc2
        self.phonopy.run_total_dos()
        self.phonopy.write_total_dos(filename=filename)

    def save_results(self, args):

        path_log = "./sscha/" + str(self.properties.temperature) + "/"
        os.makedirs(path_log, exist_ok=True)
        save_sscha_yaml(
            self._unitcell,
            self._supercell_matrix,
            self.logs,
            args,
            filename=path_log + "sscha_results.yaml",
        )
        write_fc2_to_hdf5(self.force_constants, filename=path_log + "fc2.hdf5")
        self._write_dos(filename=path_log + "total_dos.dat")

        if self._verbose:
            freq = self._run_frequencies(qmesh=args.mesh)
            print("-------- sscha runs finished --------")
            print("Temperature:      ", self.properties.temperature)
            print("Free energy:      ", self.properties.free_energy)
            print("Convergence:      ", self.properties.converge)
            print("Frequency (min):  ", "{:.6f}".format(np.min(freq)))
            print("Frequency (max):  ", "{:.6f}".format(np.max(freq)))

    @property
    def properties(self) -> PolymlpDataSSCHA:
        """Return SSCHA results."""
        return self._sscha_log[-1]

    @property
    def logs(self) -> list[PolymlpDataSSCHA]:
        """Return SSCHA progress."""
        return self._sscha_log

    @property
    def n_fc_basis(self) -> int:
        """Number of FC basis vectors."""
        return self._symfc.basis_set[2].basis_set.shape[1]

    @property
    def force_constants(self) -> np.ndarray:
        """Return FC2, shape=(n_atom, n_atom, 3, 3)."""
        return self.fc2

    @force_constants.setter
    def force_constants(self, fc2: np.ndarray):
        """Set FC2, shape=(n_atom, n_atom, 3, 3)."""
        assert fc2.shape[0] == fc2.shape[1] == self._n_atom
        assert fc2.shape[2] == fc2.shape[3] == 3
        self.fc2 = fc2


def run_sscha(
    unitcell: PolymlpStructure,
    supercell_matrix: np.ndarray,
    args,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    verbose: bool = True,
):
    """Run sscha iterations.

    Parameters
    ----------
    unitcell: Unitcell in PolymlpStructure format
    supercell_matrix: Supercell matrix.
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties object.
    args: Parameters for SSCHA.

    Attributes of args
    ------------------
    mesh: q-point mesh for computing harmonic properties using effective FC2.
    n_samples_init: Number of samples in first SSCHA loop.
    n_samples_final: Number of samples in second SSCHA loop.
    temperatures: Temperatures.
    tol: Convergence tolerance for FCs.
    max_iter: Maximum number of iterations.
    mixing: Mixing parameter.
            FCs are updated by FC2 = FC2(new) * mixing + FC2(old) * (1-mixing).
    """

    sscha = PolymlpSSCHA(
        unitcell,
        supercell_matrix,
        pot=pot,
        params=params,
        coeffs=coeffs,
        properties=properties,
        verbose=verbose,
    )

    sscha.set_initial_force_constants(algorithm=args.init, filename=args.init_file)
    freq = sscha._run_frequencies(qmesh=args.mesh)
    if verbose:
        print("Frequency (min):  ", "{:.6f}".format(np.min(freq)))
        print("Frequency (max):  ", "{:.6f}".format(np.max(freq)))
        print("Number of FC2 basis vectors:", sscha.n_fc_basis)

    for temp in args.temperatures:
        if verbose:
            print("************** Temperature:", temp, "**************")
        sscha.run(
            t=temp,
            n_samples=args.n_samples_init,
            qmesh=args.mesh,
            max_iter=args.max_iter,
            tol=args.tol,
            mixing=args.mixing,
        )
        if verbose:
            print("Increasing number of samples.")
        sscha.run(
            t=temp,
            n_samples=args.n_samples_final,
            qmesh=args.mesh,
            max_iter=args.max_iter,
            tol=args.tol,
            mixing=args.mixing,
            initialize_history=False,
        )
        sscha.save_results(args)


if __name__ == "__main__":

    import argparse
    import signal

    from pypolymlp.calculator.sscha.sscha_utils import (
        Restart,
        n_samples_setting,
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
        unitcell = Poscar(args.poscar).structure
        supercell_matrix = np.diag(args.supercell)
    elif args.yaml is not None:
        res = Restart(args.yaml)
        unitcell = res.unitcell
        supercell_matrix = res.supercell_matrix
        if args.pot is None:
            args.pot = res.mlp

    n_atom = len(unitcell.elements) * np.linalg.det(supercell_matrix)
    args = temperature_setting(args)
    args = n_samples_setting(args, n_atom)

    print_parameters(supercell_matrix, args)
    print_structure(unitcell)

    run_sscha(unitcell, supercell_matrix, args, pot=args.pot)
