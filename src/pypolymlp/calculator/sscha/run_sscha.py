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
from pypolymlp.calculator.sscha.sscha_utils import (
    PolymlpDataSSCHA,
    SSCHAParameters,
    is_imaginary,
    save_sscha_yaml,
)
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
        nac_params: Optional[dict] = None,
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

        self.phonopy = Phonopy(
            structure_to_phonopy_cell(unitcell),
            supercell_matrix,
            nac_params=nac_params,
        )
        self.n_unitcells = int(round(np.linalg.det(supercell_matrix)))
        self._n_atom = len(self.phonopy.supercell.masses)

        self.supercell_polymlp = phonopy_cell_to_structure(self.phonopy.supercell)
        self.supercell_polymlp.masses = self.phonopy.supercell.masses
        self.supercell_polymlp.supercell_matrix = supercell_matrix
        self.supercell_polymlp.n_unitcells = self.n_unitcells

        self._symfc = Symfc(
            self.phonopy.supercell,
            use_mkl=True,
            log_level=self._verbose,
        )
        self._symfc.compute_basis_set(2)
        n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
        if self._verbose and n_coeffs < 1000:
            self._symfc._log_level = 0

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
            entropy=self.ph_recip.entropy,  # J/K/mol
            harmonic_heat_capacity=self.ph_recip.heat_capacity,  # J/K/mol
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
            print("Running symfc solver.", flush=True)
        fc2_new = self._run_solver_fc2()
        self.fc2 = fc2_new * mixing + self.fc2 * (1 - mixing)
        self._sscha_current.delta = self._convergence_score(self.fc2, fc2_new)
        self._sscha_log.append(self._sscha_current)
        return self.fc2

    def _print_progress(self):

        disp_norms = np.linalg.norm(self.ph_real.displacements, axis=1)

        print(
            "temperature:      ",
            "{:.1f}".format(self._sscha_current.temperature),
            flush=True,
        )
        print("number of samples:", disp_norms.shape[0], flush=True)
        print(
            "convergence score:      ",
            "{:.6f}".format(self._sscha_current.delta),
            flush=True,
        )
        print("displacements:")
        print(
            "  average disp. (Ang.): ", "{:.6f}".format(np.mean(disp_norms)), flush=True
        )
        print(
            "  max disp. (Ang.):     ", "{:.6f}".format(np.max(disp_norms)), flush=True
        )
        print("thermodynamic_properties:", flush=True)
        print(
            "  free energy (harmonic, kJ/mol)  :",
            "{:.6f}".format(self._sscha_current.harmonic_free_energy),
            flush=True,
        )
        print(
            "  free energy (anharmonic, kJ/mol):",
            "{:.6f}".format(self._sscha_current.anharmonic_free_energy),
            flush=True,
        )
        print(
            "  free energy (sscha, kJ/mol)     :",
            "{:.6f}".format(self._sscha_current.free_energy),
            flush=True,
        )

    def set_initial_force_constants(self, algorithm="harmonic", filename=None):
        """Set initial FC2."""

        if algorithm == "harmonic":
            if self._verbose:
                print("Initial FCs: Harmonic", flush=True)
            self.fc2 = self.ph_recip.produce_harmonic_force_constants()
        elif algorithm == "const":
            if self._verbose:
                print("Initial FCs: Constants", flush=True)
            n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
            coeffs_fc2 = np.ones(n_coeffs) * 10
            coeffs_fc2[1::2] *= -1
            self.fc2 = self._recover_fc2(coeffs_fc2)
        elif algorithm == "random":
            if self._verbose:
                print("Initial FCs: Random", flush=True)
            n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
            coeffs_fc2 = (np.random.rand(n_coeffs) - 0.5) * 20
            self.fc2 = self._recover_fc2(coeffs_fc2)
        elif algorithm == "file":
            if self._verbose:
                print("Initial FCs: File", filename, flush=True)
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
                print("----------- Iteration :", n_iter, "-----------", flush=True)
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
        self._sscha_current = self._compute_sscha_properties(t=t, qmesh=qmesh)

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
        """Save SSCHA results."""
        path_log = "./sscha/" + str(self.properties.temperature) + "/"
        os.makedirs(path_log, exist_ok=True)
        freq = self._run_frequencies(qmesh=args.mesh)
        self.properties.imaginary = is_imaginary(freq)
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
            print("-------- sscha runs finished --------", flush=True)
            print("Temperature:      ", self.properties.temperature, flush=True)
            print("Free energy:      ", self.properties.free_energy, flush=True)
            print("Convergence:      ", self.properties.converge, flush=True)
            print("Frequency (min):  ", "{:.6f}".format(np.min(freq)), flush=True)
            print("Frequency (max):  ", "{:.6f}".format(np.max(freq)), flush=True)

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
    sscha_params: SSCHAParameters,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    verbose: bool = False,
):
    """Run sscha iterations.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParameters.
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties instance.
    """
    sscha = PolymlpSSCHA(
        sscha_params.unitcell,
        sscha_params.supercell_matrix,
        pot=pot,
        params=params,
        coeffs=coeffs,
        properties=properties,
        nac_params=sscha_params.nac_params,
        verbose=verbose,
    )
    sscha.set_initial_force_constants(
        algorithm=sscha_params.init_fc_algorithm,
        filename=sscha_params.init_fc_file,
    )
    freq = sscha._run_frequencies(qmesh=sscha_params.mesh)
    if verbose:
        print("Frequency (min):  ", "{:.6f}".format(np.min(freq)), flush=True)
        print("Frequency (max):  ", "{:.6f}".format(np.max(freq)), flush=True)
        print("Number of FC2 basis vectors:", sscha.n_fc_basis, flush=True)

    for temp in sscha_params.temperatures:
        if verbose:
            print("************** Temperature:", temp, "**************", flush=True)
        sscha.run(
            t=temp,
            n_samples=sscha_params.n_samples_init,
            qmesh=sscha_params.mesh,
            max_iter=sscha_params.max_iter,
            tol=sscha_params.tol,
            mixing=sscha_params.mixing,
        )
        if verbose:
            print("Increasing number of samples.", flush=True)
        sscha.run(
            t=temp,
            n_samples=sscha_params.n_samples_final,
            qmesh=sscha_params.mesh,
            max_iter=sscha_params.max_iter,
            tol=sscha_params.tol,
            mixing=sscha_params.mixing,
            initialize_history=False,
        )
        # sscha.save_results(args)
