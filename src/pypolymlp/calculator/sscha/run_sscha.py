"""Class for performing SSCHA."""

import copy
import os
from typing import Optional

import numpy as np
from phono3py.file_IO import read_fc2_from_hdf5, write_fc2_to_hdf5
from phonopy import Phonopy
from symfc import Symfc

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.harmonic_reciprocal import HarmonicReciprocal
from pypolymlp.calculator.sscha.sscha_params import SSCHAParameters
from pypolymlp.calculator.sscha.sscha_utils import (
    PolymlpDataSSCHA,
    is_imaginary,
    save_sscha_yaml,
)
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)


class SSCHA:
    """Class for performing SSCHA."""

    def __init__(
        self,
        sscha_params: SSCHAParameters,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        sscha_params: Structures and SSCHA parameters in SSCHAParameters format.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        self._verbose = verbose
        if properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._unitcell = sscha_params.unitcell
        self._supercell_matrix = sscha_params.supercell_matrix
        self._sscha_params = sscha_params

        self.phonopy = Phonopy(
            structure_to_phonopy_cell(self._unitcell),
            self._supercell_matrix,
            nac_params=sscha_params.nac_params,
        )
        self._n_atom = len(self.phonopy.supercell.masses)
        self.n_unitcells = round(np.linalg.det(self._supercell_matrix))

        self.supercell_polymlp = phonopy_cell_to_structure(self.phonopy.supercell)
        self.supercell_polymlp.masses = self.phonopy.supercell.masses
        self.supercell_polymlp.supercell_matrix = self._supercell_matrix
        self.supercell_polymlp.n_unitcells = self.n_unitcells
        self._sscha_params.supercell = self.supercell_polymlp

        cutoff = {2: sscha_params.cutoff_radius}
        self._symfc = Symfc(
            self.phonopy.supercell,
            cutoff=cutoff,
            use_mkl=True,
            log_level=self._verbose,
        )
        self._symfc.compute_basis_set(2)

        n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
        if self._verbose and n_coeffs < 1000:
            self._symfc._log_level = 0

        if self._sscha_params.n_samples_init is None:
            self._sscha_params.set_n_samples_from_basis(n_coeffs)
            if self._verbose:
                print("Number of supercells is automatically determined.")
                print("- first loop:", self._sscha_params.n_samples_init)
                print("- second loop:", self._sscha_params.n_samples_final)

        self.ph_real = HarmonicReal(self.supercell_polymlp, self._prop)
        self.ph_recip = HarmonicReciprocal(self.phonopy, self._prop)

        self._fc2 = None
        self._sscha_current = None
        self._sscha_log = []

    def set_initial_force_constants(self, fc2: Optional[np.ndarray] = None):
        """Set initial FC2."""
        if fc2 is not None:
            if self._verbose:
                print("Initial FCs: Numpy array", flush=True)
            self._fc2 = fc2
            return self

        if self._sscha_params.init_fc_algorithm == "harmonic":
            if self._verbose:
                print("Initial FCs: Harmonic", flush=True)
            self._fc2 = self.ph_recip.produce_harmonic_force_constants()
        elif self._sscha_params.init_fc_algorithm == "const":
            if self._verbose:
                print("Initial FCs: Constants", flush=True)
            n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
            coeffs_fc2 = np.ones(n_coeffs) * 10
            coeffs_fc2[1::2] *= -1
            self._fc2 = self._recover_fc2(coeffs_fc2)
        elif self._sscha_params.init_fc_algorithm == "random":
            if self._verbose:
                print("Initial FCs: Random", flush=True)
            n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
            coeffs_fc2 = (np.random.rand(n_coeffs) - 0.5) * 20
            self._fc2 = self._recover_fc2(coeffs_fc2)
        elif self._sscha_params.init_fc_algorithm == "file":
            filename = self._sscha_params.init_fc_file
            if self._verbose:
                print("Initial FCs: File", filename, flush=True)
            self._fc2 = read_fc2_from_hdf5(filename)
        return self

    def run_frequencies(self, qmesh: Optional[tuple] = None):
        """Calculate effective phonon frequencies from FC2."""
        if qmesh is None:
            qmesh = self._sscha_params.mesh
        self.phonopy.force_constants = self._fc2
        self.phonopy.run_mesh(qmesh)
        mesh_dict = self.phonopy.get_mesh_dict()
        return mesh_dict["frequencies"]

    def _unit_kjmol(self, e):
        """Convert energy in eV/supercell to energy in kJ/mol."""
        return e * EVtoKJmol / self.n_unitcells

    def _compute_sscha_properties(self, t: float = 1000):
        """Compute SSCHA properties using FC2."""
        if self._verbose:
            print("Computing SSCHA properties from FC2.", flush=True)
        qmesh = self._sscha_params.mesh
        self.ph_recip.force_constants = self._fc2
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
            static_forces=self.ph_real.static_forces,  # eV/ang
            average_forces=self.ph_real.average_forces,  # eV/ang
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
        """Compute convergence score."""
        norm1 = np.linalg.norm(fc2_update - fc2_init)
        norm2 = np.linalg.norm(fc2_init)
        return norm1 / norm2

    def _single_iter(self, t: float = 1000, n_samples: int = 100) -> np.ndarray:
        """Run a standard single sscha iteration."""
        self.ph_real.force_constants = self._fc2
        self.ph_real.run(t=t, n_samples=n_samples, eliminate_outliers=True)
        self._sscha_current = self._compute_sscha_properties(t=t)

        if self._verbose:
            print("Running symfc solver.", flush=True)
        fc2_new = self._run_solver_fc2()
        mixing = self._sscha_params.mixing
        self._fc2 = fc2_new * mixing + self._fc2 * (1 - mixing)
        self._sscha_current.delta = self._convergence_score(self._fc2, fc2_new)
        self._sscha_log.append(self._sscha_current)
        return self._fc2

    def _print_progress(self):
        """Print progress in SSCHA iterations."""
        disp_norms = np.linalg.norm(self.ph_real.displacements, axis=1)

        print(
            "temperature:      ",
            "{:.1f}".format(self._sscha_current.temperature),
            flush=True,
        )
        print("number of samples:", disp_norms.shape[0], flush=True)
        print(
            "convergence score:     ",
            "{:.6f}".format(self._sscha_current.delta),
            flush=True,
        )
        print("displacements:")
        print("- average disp. (Ang.):", np.round(np.mean(disp_norms), 6), flush=True)
        print("- max disp. (Ang.):    ", np.round(np.max(disp_norms), 6), flush=True)
        print("thermodynamic_properties:", flush=True)
        print(
            "- free energy (harmonic, kJ/mol)  :",
            "{:.6f}".format(self._sscha_current.harmonic_free_energy),
            flush=True,
        )
        print(
            "- free energy (anharmonic, kJ/mol):",
            "{:.6f}".format(self._sscha_current.anharmonic_free_energy),
            flush=True,
        )
        print(
            "- free energy (sscha, kJ/mol)     :",
            "{:.6f}".format(self._sscha_current.free_energy),
            flush=True,
        )

    def precondition(
        self,
        t: float = 1000,
        n_samples: int = 100,
        max_iter: int = 10,
        tol: float = 0.01,
    ):
        """Precondition sscha iterations.

        Parameters
        ----------
        t: Temperature (K).
        """
        if self._fc2 is None:
            self._fc2 = self.set_initial_force_constants()

        n_iter, delta = 1, 1e10
        while n_iter <= max_iter and delta > tol:
            if self._verbose:
                txt = "--------------- Iteration : " + str(n_iter) + " ---------------"
                print(txt, flush=True)
            self._fc2 = self._single_iter(t=t, n_samples=n_samples)
            if self._verbose:
                self._print_progress()

            delta = self._sscha_current.delta
            n_iter += 1

    def run(
        self,
        t: float = 1000,
        accurate: bool = False,
        initialize_history: bool = True,
    ):
        """Run sscha iterations.

        Parameters
        ----------
        t: Temperature (K).
        accurate: Increase number of sample supercells.
        initialize_history: Initialize history logs.
        """
        if self._fc2 is None:
            self._fc2 = self.set_initial_force_constants()

        if accurate:
            n_samples = self._sscha_params.n_samples_final
        else:
            n_samples = self._sscha_params.n_samples_init
        if initialize_history:
            self._sscha_log = []

        n_iter, delta = 1, 1e10
        while n_iter <= self._sscha_params.max_iter and delta > self._sscha_params.tol:
            if self._verbose:
                txt = "--------------- Iteration : " + str(n_iter) + " ---------------"
                print(txt, flush=True)
            self._fc2 = self._single_iter(t=t, n_samples=n_samples)
            if self._verbose:
                self._print_progress()

            delta = self._sscha_current.delta
            n_iter += 1

        converge = True if delta < self._sscha_params.tol else False
        self._sscha_log[-1].converge = converge
        self._sscha_current = self._compute_sscha_properties(t=t)

    def _write_dos(self, filename: str = "total_dos.dat"):
        """Save phonon DOS file."""
        self.phonopy.force_constants = self._fc2
        self.phonopy.run_total_dos()
        self.phonopy.write_total_dos(filename=filename)

    def save_results(self):
        """Save SSCHA results for current temperature."""
        path_log = "./sscha/" + str(self.properties.temperature) + "/"
        os.makedirs(path_log, exist_ok=True)
        freq = self.run_frequencies(qmesh=self._sscha_params.mesh)
        self.properties.imaginary = is_imaginary(freq)
        save_sscha_yaml(
            self._sscha_params,
            self.logs,
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
    def sscha_params(self) -> SSCHAParameters:
        """Return SSCHA parameters."""
        return self._sscha_params

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
        return self._fc2

    @force_constants.setter
    def force_constants(self, fc2: np.ndarray):
        """Set FC2, shape=(n_atom, n_atom, 3, 3)."""
        assert fc2.shape[0] == fc2.shape[1] == self._n_atom
        assert fc2.shape[2] == fc2.shape[3] == 3
        self._fc2 = fc2


def _run_precondition(
    sscha: SSCHA,
    sscha_params: SSCHAParameters,
    verbose: bool = False,
):
    """Run a procedure to perform precondition."""

    if verbose:
        print("---", flush=True)
        print("Preconditioning.", flush=True)
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)

    n_samples = max(min(sscha_params.n_samples_init // 50, 100), 5)
    sscha.precondition(
        t=sscha_params.temperatures[0],
        n_samples=n_samples,
        tol=0.01,
        max_iter=5,
    )
    if sscha_params.tol < 0.003:
        n_samples = max(min(sscha_params.n_samples_init // 10, 500), 10)
        sscha.precondition(
            t=sscha_params.temperatures[0],
            n_samples=n_samples,
            tol=0.005,
            max_iter=5,
        )
    return sscha


def _run_target_sscha(
    sscha: SSCHA,
    sscha_params: SSCHAParameters,
    verbose: bool = False,
):
    """Run SSCHA for target temperatures."""
    for temp in sscha_params.temperatures:
        if verbose:
            print("************** Temperature:", temp, "**************", flush=True)
            print("Increasing number of samples.", flush=True)
        sscha.run(t=temp, accurate=False)
        if verbose:
            print("Increasing number of samples.", flush=True)
        sscha.run(t=temp, accurate=True, initialize_history=False)
        sscha.save_results()
    return sscha


def run_sscha(
    sscha_params: SSCHAParameters,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    fc2: Optional[np.ndarray] = None,
    precondition: bool = True,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParameters.
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties instance.
    """
    sscha = SSCHA(
        sscha_params,
        pot=pot,
        params=params,
        coeffs=coeffs,
        properties=properties,
        verbose=verbose,
    )
    sscha.set_initial_force_constants(fc2=fc2)
    freq = sscha.run_frequencies()
    if verbose:
        print("Frequency (min):      ", np.round(np.min(freq), 5), flush=True)
        print("Frequency (max):      ", np.round(np.max(freq), 5), flush=True)

    if precondition:
        sscha = _run_precondition(sscha, sscha_params, verbose=verbose)

    if verbose:
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)
    sscha = _run_target_sscha(sscha, sscha_params, verbose=verbose)
    return sscha


def run_sscha_large_system(
    sscha_params: SSCHAParameters,
    pot: Optional[str] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[np.ndarray] = None,
    properties: Optional[Properties] = None,
    fc2: Optional[np.ndarray] = None,
    precondition: bool = True,
    verbose: bool = False,
):
    """Run sscha iterations for multiple temperatures using cutoff temporarily.

    Parameters
    ----------
    sscha_params: Parameters for SSCHA in SSCHAParameters.
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties instance.
    """
    sscha_params_target = copy.deepcopy(sscha_params)
    if sscha_params.cutoff_radius is None or sscha_params.cutoff_radius > 7.0:
        sscha_params.cutoff_radius = 6.0
        rerun = True
    else:
        rerun = False

    sscha = SSCHA(
        sscha_params,
        pot=pot,
        params=params,
        coeffs=coeffs,
        properties=properties,
        verbose=verbose,
    )
    sscha.set_initial_force_constants(fc2=fc2)
    freq = sscha.run_frequencies()
    if verbose:
        print("Frequency (min):      ", np.round(np.min(freq), 5), flush=True)
        print("Frequency (max):      ", np.round(np.max(freq), 5), flush=True)

    if precondition:
        sscha = _run_precondition(sscha, sscha_params, verbose=verbose)

    if rerun:
        if verbose:
            print("---", flush=True)
            print("Run SSCHA with temporal cutoff.", flush=True)
            print("Temporal cutoff radius:", sscha_params.cutoff_radius, flush=True)
            print("Size of FC2 basis-set: ", sscha.n_fc_basis, flush=True)
        sscha.run(t=sscha_params.temperatures[0], accurate=False)
        fc2_rerun = sscha.force_constants
        sscha_params.cutoff_radius = sscha_params_target.cutoff_radius

        sscha = SSCHA(
            sscha_params_target,
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            verbose=verbose,
        )
        sscha.set_initial_force_constants(fc2=fc2_rerun)

    if verbose:
        print("Size of FC2 basis-set:", sscha.n_fc_basis, flush=True)

    sscha = _run_target_sscha(sscha, sscha_params, verbose=verbose)
    return sscha
