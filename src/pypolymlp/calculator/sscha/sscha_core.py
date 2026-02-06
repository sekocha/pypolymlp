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
from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.calculator.sscha.sscha_io import save_sscha_yaml
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)


class SSCHACore:
    """Class for performing SSCHA."""

    def __init__(
        self,
        sscha_params: SSCHAParams,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        sscha_params: Parameters for SSCHA and structures in SSCHAParams.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.
        verbose: Verbose mode.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        self._verbose = verbose
        if properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._phonopy = Phonopy(
            structure_to_phonopy_cell(sscha_params.unitcell),
            sscha_params.supercell_matrix,
            nac_params=sscha_params.nac_params,
        )
        self._sscha_params = sscha_params
        self._n_atom = sscha_params.n_atom
        self._n_unitcells = sscha_params.n_unitcells
        self._n_coeffs = None
        self._fc2 = None
        self._data_current = None
        self._sscha_log = []

        self._symfc = self._set_symfc(sscha_params.cutoff_radius)
        self._set_num_samples()
        self._ph_real, self._ph_recip = self._set_harmonic_calculators()

    def _set_symfc(self, cutoff_radius: Optional[float] = None):
        """Initialize Symfc instance."""
        cutoff = {2: cutoff_radius}
        sup = self._phonopy.supercell
        self._symfc = Symfc(sup, cutoff=cutoff, use_mkl=True, log_level=self._verbose)
        self._symfc.compute_basis_set(2)

        self._n_coeffs = self._symfc.basis_set[2].basis_set.shape[1]
        if self._verbose and self._n_coeffs < 1000:
            self._symfc._log_level = 0
        return self._symfc

    def _set_num_samples(self):
        """Initialize number of supercell samples."""
        if self._sscha_params.n_samples_init is None:
            self._sscha_params.set_n_samples_from_basis(self._n_coeffs)
            if self._verbose:
                print("Number of supercells is automatically determined.")
                print("- first loop:", self._sscha_params.n_samples_init)
                print("- second loop:", self._sscha_params.n_samples_final)
        return self

    def _set_harmonic_calculators(self):
        """Initialize calculators for harmonic properties."""
        supercell_polymlp = phonopy_cell_to_structure(self._phonopy.supercell)
        supercell_polymlp.masses = self._phonopy.supercell.masses
        supercell_matrix = self._sscha_params.supercell_matrix
        supercell_polymlp.supercell_matrix = supercell_matrix
        supercell_polymlp.n_unitcells = self._n_unitcells
        self._sscha_params.supercell = supercell_polymlp

        self._ph_real = HarmonicReal(supercell_polymlp, self._prop)
        self._ph_recip = HarmonicReciprocal(self._phonopy, self._prop)
        return self._ph_real, self._ph_recip

    def set_initial_force_constants(self, fc2: Optional[np.ndarray] = None):
        """Set initial FC2."""
        if fc2 is not None:
            if self._verbose:
                print("Initial FCs: Numpy array", flush=True)
            self._fc2 = fc2
            return self

        algorithm = self._sscha_params.init_fc_algorithm
        if algorithm not in ("harmonic", "const", "random", "file"):
            raise RuntimeError("Available method for initial FCs not given.")

        if algorithm == "harmonic":
            if self._verbose:
                print("Initial FCs: Harmonic", flush=True)
            self._fc2 = self._ph_recip.produce_harmonic_force_constants()
        elif algorithm == "const":
            if self._verbose:
                print("Initial FCs: Constants", flush=True)
            coeffs_fc2 = np.ones(self._n_coeffs) * 10
            coeffs_fc2[1::2] *= -1
            self._fc2 = self._recover_fc2(coeffs_fc2)
        elif algorithm == "random":
            if self._verbose:
                print("Initial FCs: Random", flush=True)
            coeffs_fc2 = (np.random.rand(self._n_coeffs) - 0.5) * 20
            self._fc2 = self._recover_fc2(coeffs_fc2)
        elif algorithm == "file":
            filename = self._sscha_params.init_fc_file
            if self._verbose:
                print("Initial FCs: File", filename, flush=True)
            self._fc2 = read_fc2_from_hdf5(filename)
        return self

    def run_frequencies(self, qmesh: Optional[tuple] = None):
        """Calculate effective phonon frequencies from FC2."""
        if qmesh is None:
            qmesh = self._sscha_params.mesh
        self._phonopy.force_constants = self._fc2
        self._phonopy.run_mesh(qmesh)
        mesh_dict = self._phonopy.get_mesh_dict()
        return mesh_dict["frequencies"]

    def _compute_sscha_properties(self, temp: float = 1000):
        """Compute SSCHA properties using FC2."""
        if self._verbose:
            print("Computing SSCHA properties from FC2.", flush=True)

        qmesh = self._sscha_params.mesh
        self._ph_recip.force_constants = self._fc2
        self._ph_recip.compute_thermal_properties(temp=temp, qmesh=qmesh)

        data = SSCHAData(
            temperature=temp,
            static_potential=self._ph_real.static_potential,  # kJ/mol
            harmonic_potential=self._ph_real.average_harmonic_potential,  # kJ/mol
            harmonic_free_energy=self._ph_recip.free_energy,  # kJ/mol
            average_potential=self._ph_real.average_full_potential,  # kJ/mol
            anharmonic_free_energy=self._ph_real.average_anharmonic_potential,
            entropy=self._ph_recip.entropy,  # J/K/mol
            harmonic_heat_capacity=self._ph_recip.heat_capacity,  # J/K/mol
            static_forces=self._ph_real.static_forces,  # eV/ang
            average_forces=self._ph_real.average_forces,  # eV/ang
        )
        return data

    def _run_solver_fc2(self):
        """Estimate FC2 from a forces-displacements dataset."""
        self._symfc.displacements = self._ph_real.displacements.transpose((0, 2, 1))
        self._symfc.forces = self._ph_real.forces.transpose((0, 2, 1))
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

    def _single_iter(self, temp: float = 1000, n_samples: int = 100) -> np.ndarray:
        """Run a standard single sscha iteration."""
        self._ph_real.force_constants = self._fc2
        self._ph_real.run(temp=temp, n_samples=n_samples, eliminate_outliers=True)
        self._data_current = self._compute_sscha_properties(temp=temp)

        if self._verbose:
            print("Running symfc solver.", flush=True)
        fc2_new = self._run_solver_fc2()
        mixing = self._sscha_params.mixing
        self._fc2 = fc2_new * mixing + self._fc2 * (1 - mixing)

        self._data_current.delta = self._convergence_score(self._fc2, fc2_new)
        self._sscha_log.append(self._data_current)
        return self._fc2

    def _final_iter(self, temp: float = 1000, n_samples: int = 100) -> np.ndarray:
        """Run the final iteration."""
        self._ph_real.force_constants = self._fc2
        self._ph_real.run(temp=temp, n_samples=n_samples, eliminate_outliers=True)
        self._data_current = self._compute_sscha_properties(temp=temp)
        self._sscha_log.append(self._data_current)
        return self

    def _is_imaginary(self, tol: float = -0.01):
        """Check if imaginary frequencies exist only using frequency data."""
        freq = self.run_frequencies(qmesh=self._sscha_params.mesh)
        n_imag = np.count_nonzero(freq < tol)
        return (n_imag / freq.size) > 1e-5

    def _print_separator(self, n_iter: int):
        """Print separator between SSCHA iterations."""
        txt = "------------------ Iteration : " + str(n_iter) + " ------------------"
        print(txt, flush=True)

    def _print_progress(self):
        """Print progress in SSCHA iterations."""
        disp_norms = np.linalg.norm(self._ph_real.displacements, axis=1)

        data = self._data_current
        print("temperature:          ", "{:.1f}".format(data.temperature), flush=True)
        print("number of samples:    ", disp_norms.shape[0], flush=True)
        print("convergence score:    ", "{:.6f}".format(data.delta), flush=True)
        print("displacements:")
        print("- average disp. (Ang.):", np.round(np.mean(disp_norms), 6), flush=True)
        print("- max disp. (Ang.):    ", np.round(np.max(disp_norms), 6), flush=True)

        print("thermodynamic_properties:", flush=True)
        prefix = "- free energy (harmonic, kJ/mol):   "
        print(prefix, "{:.6f}".format(data.harmonic_free_energy), flush=True)
        prefix = "- free energy (anharmonic, kJ/mol): "
        print(prefix, "{:.6f}".format(data.anharmonic_free_energy), flush=True)
        prefix = "- free energy (sscha, kJ/mol):      "
        print(prefix, "{:.6f}".format(data.free_energy), flush=True)

    def precondition(
        self,
        temp: float = 1000,
        n_samples: int = 100,
        max_iter: int = 10,
        tol: float = 0.01,
    ):
        """Precondition sscha iterations.

        Parameters
        ----------
        temp: Temperature (K).
        """
        if self._fc2 is None:
            self._fc2 = self.set_initial_force_constants()

        n_iter, delta = 1, 1e10
        while n_iter <= max_iter and delta > tol:
            if self._verbose:
                self._print_separator(n_iter)
            self._fc2 = self._single_iter(temp=temp, n_samples=n_samples)
            if self._verbose:
                self._print_progress()

            delta = self._data_current.delta
            n_iter += 1

        return self

    def run(self, temp: float = 1000, initialize_history: bool = True):
        """Run sscha iterations.

        Parameters
        ----------
        t: Temperature (K).
        initialize_history: Initialize history logs.
        """
        if self._fc2 is None:
            self._fc2 = self.set_initial_force_constants()

        if initialize_history:
            self._sscha_log = []

        n_iter, delta = 1, 1e10
        max_iter, tol = self._sscha_params.max_iter, self._sscha_params.tol
        n_samples = self._sscha_params.n_samples_init
        while (n_iter <= max_iter and delta > tol) or n_iter < 3:
            if self._verbose:
                self._print_separator(n_iter)
            self._fc2 = self._single_iter(temp=temp, n_samples=n_samples)
            if self._verbose:
                self._print_progress()

            delta = self._data_current.delta
            n_iter += 1

        converge = True if delta < tol else False
        if not converge:
            self._data_current.converge = converge
            if self._verbose:
                print("Error: SSCHA calculation not converge.", flush=True)
            return self

        if self._verbose:
            self._print_separator(n_iter)
            print("SSCHA calculation converges.", flush=True)
            print("Proceeding to a more accurate evaluation.", flush=True)

        self._final_iter(temp=temp, n_samples=self._sscha_params.n_samples_final)
        self._data_current.converge = converge
        self._data_current.delta = delta
        self._data_current.imaginary = self._is_imaginary()
        if self._verbose:
            self._print_progress()

        return self

    def _write_dos(
        self,
        filename: str = "total_dos.dat",
        write_pdos: bool = False,
        qmesh: Optional[tuple] = None,
    ):
        """Save phonon DOS file."""
        self._phonopy.force_constants = self._fc2
        self._phonopy.run_total_dos()
        self._phonopy.write_total_dos(filename=filename)
        if write_pdos:
            if qmesh is None:
                qmesh = self._sscha_params.mesh
            self._phonopy.run_mesh(
                qmesh, is_mesh_symmetry=False, with_eigenvectors=True
            )
            path = "/".join(filename.split("/")[:-1])
            self._phonopy.run_projected_dos()
            self._phonopy.write_projected_dos(filename=path + "/projected_dos.dat")

    def _print_final_results(self):
        """Print SSCHA results for current temperature."""
        data = self._data_current
        freq = self.run_frequencies(qmesh=self._sscha_params.mesh)
        print("---------------- sscha runs finished ----------------", flush=True)
        print("Temperature:      ", data.temperature, flush=True)
        print("Free energy:      ", data.free_energy, flush=True)
        print("Convergence:      ", data.converge, flush=True)
        print("Frequency (min):  ", "{:.6f}".format(np.min(freq)), flush=True)
        print("Frequency (max):  ", "{:.6f}".format(np.max(freq)), flush=True)

    def save_results(self, path: str = "./sscha", write_pdos: bool = False):
        """Save SSCHA results for current temperature."""
        temp = self._data_current.temperature
        path_log = path + "/" + str(temp) + "/"
        os.makedirs(path_log, exist_ok=True)
        filename = path_log + "sscha_results.yaml"
        save_sscha_yaml(self._sscha_params, self.logs, filename=filename)
        write_fc2_to_hdf5(self.force_constants, filename=path_log + "fc2.hdf5")
        self._write_dos(filename=path_log + "total_dos.dat", write_pdos=write_pdos)
        if self._verbose:
            self._print_final_results()

        return self

    @property
    def sscha_params(self) -> SSCHAParams:
        """Return SSCHA parameters."""
        return self._sscha_params

    @property
    def properties(self) -> SSCHAData:
        """Return SSCHA results."""
        return self._data_current

    @property
    def logs(self) -> list[SSCHAData]:
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
