"""Class for harmonic contribution in reciprocal space."""

from typing import Optional

import numpy as np
import scipy
from phonopy import Phonopy

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_utils import frequencies_from_eigvals

from .harmonic_reciprocal_base import HarmonicReciprocalBase


class HarmonicReciprocal(HarmonicReciprocalBase):
    """Class for harmonic contribution in reciprocal space."""

    def __init__(
        self,
        phonopy_obj: Phonopy,
        properties: Properties,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        phonopy_obj: Phonopy class object.
        property: Property class object to calculate energies and forces.
        fc2: Second-order force constants.
        """
        super().__init__(phonopy_obj, properties, fc2=fc2, verbose=verbose)

    def compute_mesh_properties(self, qmesh: tuple = (10, 10, 10)):
        """Compute mesh properties."""
        self._ph.run_mesh(qmesh)
        self._frequencies = self._ph._mesh.frequencies

        if self._verbose:
            mesh_dict = self._ph.get_mesh_dict()
            qpoints = mesh_dict["qpoints"]

            tol = 0.01
            match = np.where(self._frequencies < -tol)
            print("Imaginary frequencies (k-space):", flush=True)
            for q, w in zip(qpoints[match[0]], self._frequencies[match]):
                print(q, w, flush=True)
            print("Zero frequencies (k-space):", flush=True)
            print(self._frequencies[np.abs(self._frequencies) < tol], flush=True)

        self._ph.run_total_dos()
        self._mesh_dict = self._ph.get_mesh_dict()
        return self

    def compute_thermal_properties(
        self,
        temp: float = 1000,
        qmesh: tuple = (10, 10, 10),
        hide_imaginary: bool = False,
    ):
        """Compute thermal properties."""
        self._ph.run_mesh(qmesh)
        self._frequencies = self._ph._mesh.frequencies
        self._ph.run_thermal_properties(t_step=10, t_max=temp, t_min=temp)
        self._tp_dict = self._ph.get_thermal_properties_dict()
        free_energy_raw = self._tp_dict["free_energy"][0]

        if self._verbose:
            mesh_dict = self._ph.get_mesh_dict()
            qpoints = mesh_dict["qpoints"]

            tol = 0.01
            match = np.where(self._frequencies < -tol)
            print("Imaginary frequencies (k-space):", flush=True)
            for q, w in zip(qpoints[match[0]], self._frequencies[match]):
                print(q, w, flush=True)
            print("Zero frequencies (k-space):", flush=True)
            print(self._frequencies[np.abs(self._frequencies) < tol], flush=True)

        free_energy_rev = None
        if hide_imaginary:
            freq = self._ph._mesh._frequencies
            self._ph._mesh._frequencies = self._hide_imaginary_modes(freq)
            self._ph.run_thermal_properties(t_step=10, t_max=temp, t_min=temp)
            self._tp_dict = self._ph.get_thermal_properties_dict()
            free_energy_rev = self._tp_dict["free_energy"][0]
        return free_energy_raw, free_energy_rev


class HarmonicReciprocalReduced(HarmonicReciprocalBase):
    """Class for harmonic contribution in reciprocal space."""

    def __init__(
        self,
        phonopy_obj: Phonopy,
        properties: Properties,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        phonopy_obj: Phonopy class object.
        property: Property class object to calculate energies and forces.
        fc2: Second-order force constants.
        """
        super().__init__(phonopy_obj, properties, fc2=fc2, verbose=verbose)

    def _solve_eigen_equation(self, qpoints: np.ndarray) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        square_freqs = []
        for q in qpoints:
            D = self._ph.get_dynamical_matrix_at_q(q)
            eigvals, eigvecs = scipy.linalg.eigh(D)
            square_freqs.append(eigvals)

        square_freqs = np.array(square_freqs)
        freqs = frequencies_from_eigvals(square_freqs)
        return freqs

    def compute_mesh_properties(self, qmesh: tuple = (10, 10, 10)):
        """Compute mesh properties."""
        # TODO: Avoid solving eigenvalue problems here.
        self._ph.run_mesh(qmesh)
        # self._frequencies = self._ph._mesh.frequencies

        self._frequencies = self._solve_eigen_equation(self._ph._mesh.qpoints)
        if self._verbose:
            mesh_dict = self._ph.get_mesh_dict()
            qpoints = mesh_dict["qpoints"]

            tol = 0.01
            match = np.where(self._frequencies < -tol)
            print("Imaginary frequencies (k-space):", flush=True)
            for q, w in zip(qpoints[match[0]], self._frequencies[match]):
                print(q, w, flush=True)
            print("Zero frequencies (k-space):", flush=True)
            print(self._frequencies[np.abs(self._frequencies) < tol], flush=True)

        self._ph.run_total_dos()
        self._mesh_dict = self._ph.get_mesh_dict()
        return self
