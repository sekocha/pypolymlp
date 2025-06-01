"""Class for harmonic contribution in real space."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.fc_utils import eval_properties_fc2
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.displacements import get_structures_from_displacements
from pypolymlp.core.units import Avogadro, Kb, Planck
from pypolymlp.core.utils import mass_table

const_planck = Planck * 1e12  # 6.62607015e-22
const_sq_angfreq_to_sq_freq_thz = 2.4440020137144617e2
const_amplitude = 1.010758017933576


class HarmonicReal:
    """Class for harmonic contribution in real space.

    Constants
    ---------
    const_sq_angfreq_to_sq_freq_thz:
        1.602176634e-19 (eV->J) * 6.02214076e23 (avogadro) * 0.1 / (4 * pi^2)

    const_amplitude: J*s/THz -> atomic_mass * angstrom^2
        6.62607015e-34 * 6.02214076e23 * 1e11 / (4 * pi * pi)
    """

    def __init__(
        self,
        supercell: PolymlpStructure,
        properties: Properties,
        n_unitcells: Optional[int] = None,
        fc2: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        supercell: Supercell structure.
        properties: Properties class object to calculate energies and forces.
        n_unitcells: Number of unitcells in supercell
        fc2: Second-order force constants.
        """
        self.supercell = supercell
        self._n_atom = len(supercell.elements)
        self.fc2 = fc2
        if n_unitcells is not None:
            self.supercell.n_unitcells = n_unitcells

        self.prop = properties
        self._verbose = verbose

        self._mesh_dict = dict()
        self._tp_dict = dict()

        self._set_mass()
        self._set_inverse_axis()
        self._check_n_unitcells()

        self._e0, self._f0, _ = self.prop.eval(self.supercell)

    def _set_mass(self):
        if self.supercell.masses is None:
            table = mass_table()
            masses = [table[e] for e in self.supercell.elements]
            self.supercell.masses = masses

    def _set_inverse_axis(self):
        if self.supercell.axis_inv is None:
            self.supercell.axis_inv = np.linalg.inv(self.supercell.axis)

    def _check_n_unitcells(self):
        if self.supercell.n_unitcells is None:
            raise ValueError("Attribute n_unitcells is required for HarmonicReal.")

    def eval(self, structures: list[PolymlpStructure]) -> tuple[np.ndarray, np.ndarray]:
        """Compute energies and forces of structures.

        Parameters
        ----------
        structures: Structures.

        Return
        ------
        energies: Energies, shape=(n_str)
        forces: Forces, shape=(n_str, 3, n_atom)
        """

        energies, forces, _ = self.prop.eval_multiple(structures)
        return np.array(energies), np.array(forces)

    def _solve_eigen_equation(self) -> dict:
        """Solve eigenvalue equation for dynamical matrix."""
        fc2 = self.fc2.transpose((0, 2, 1, 3))
        size = fc2.shape[0] * fc2.shape[1]
        fc2 = np.reshape(fc2, (size, size))

        masses = np.repeat(self.supercell.masses, 3)
        masses_sqrt = np.reciprocal(np.sqrt(masses))
        dyn = (np.diag(masses_sqrt) @ fc2) @ np.diag(masses_sqrt)
        square_w, eigvecs = np.linalg.eigh(dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        negative_square_w = square_w < 0.0
        positive_square_w = square_w >= 0.0
        freq = np.zeros(square_w.shape)
        freq[positive_square_w] = np.sqrt(square_w[positive_square_w])
        freq[negative_square_w] = -np.sqrt(-square_w[negative_square_w])

        self._mesh_dict["frequencies"] = freq
        self._mesh_dict["eigenvectors"] = eigvecs
        return self._mesh_dict

    def _hide_imaginary_modes(self, freq: np.ndarray, freq_threshold: float = 0.1):
        """Mask branches with imaginary frequencies."""
        freq_rev = np.array(freq)
        freq_rev[np.where(freq_rev < freq_threshold)] = 0.0
        return freq_rev

    def _get_distribution(self, t: float = 1000, n_samples: int = 100):
        """Calculate atomic real-space distribution from density matrix."""
        freq = self._hide_imaginary_modes(self._mesh_dict["frequencies"])
        nonzero = np.isclose(freq, 0.0) == False

        beta = np.inf if np.isclose(t, 0.0) else 1.0 / (Kb * t)
        const_exp = 0.5 * beta * const_planck
        beta_h_freq = const_exp * freq[nonzero]

        # Calculate occupancies.
        occ = np.zeros(freq.shape)
        occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

        # Calculate amplitudes.
        # Arbitrary setting for branches with low and imaginary frequencies.
        amplitudes = np.ones(freq.shape) * 0.01
        rec_freq = np.array([1 / f for f in freq[nonzero]])
        amplitudes[nonzero] = const_amplitude * occ[nonzero] * rec_freq

        # Generate atomic displacements in normal coordinates.
        cov = np.diag(amplitudes)
        mean = np.zeros(cov.shape[0])
        disp_normal_coords = np.random.multivariate_normal(mean, cov, n_samples)

        # Generate atomic displacements.
        eigvecs = self._mesh_dict["eigenvectors"]
        masses_sqrt = np.sqrt(np.repeat(self.supercell.masses, 3))
        dot1 = eigvecs @ disp_normal_coords.T
        disps = (np.diag(np.reciprocal(masses_sqrt)) @ dot1).T
        disps = disps.reshape((n_samples, -1, 3)).transpose((0, 2, 1))
        return disps

    def _compute_harmonic_properties(self):
        """Calculate harmonic potentials and average forces."""

        N3 = self.fc2.shape[0] * self.fc2.shape[2]
        fc2 = self.fc2.transpose((0, 2, 1, 3)).reshape((N3, N3))

        pot_harmonic, residual_f = [], []
        for d in self._disps:
            energy, harmonic_forces = eval_properties_fc2(fc2, d.T.reshape(-1))
            pot_harmonic.append(energy)
            residual_f.append(harmonic_forces + self._f0)

        pot_harmonic = np.array(pot_harmonic)
        residual_f = np.array(residual_f)
        average_forces = np.mean(self._forces - residual_f, axis=0)
        return pot_harmonic, average_forces

    def _eliminate_outliers(self, tol_negative: float = -10):
        """Eliminate outliers."""
        energies = np.array(self._energies_full)
        ids1 = np.where(energies > tol_negative)[0]

        e_ave = np.mean(energies[ids1])
        tol = 2 * abs(e_ave)
        ids2 = np.where(np.abs(energies - e_ave) < tol)[0]
        ids = set(ids1) & set(ids2)

        if self._verbose:
            entire_ids = set(list(range(len(energies))))
            outlier_ids = entire_ids - ids
            if self._verbose and len(outlier_ids) > 0:
                print("Outliers are eliminated.")
                print("- Average potential energy:", "{:f}".format(e_ave))
                for i in sorted(outlier_ids):
                    print(
                        "- Potential energy (outlier " + str(i) + "):",
                        "{:f}".format(energies[i]),
                    )

        ids = np.array(list(ids))
        self._disps = self._disps[ids]
        self._supercells = [self._supercells[i] for i in ids]
        self._forces = self._forces[ids]
        self._energies_full = self._energies_full[ids]
        return self

    def run(self, t: int = 1000, n_samples: int = 100, eliminate_outliers: bool = True):
        """Run harmonic real-space part of SSCHA.

        Parameters
        ----------
        t: Temperature (K).
        n_samples: Number of sample structures.
        eliminate_outliers: Eliminate structures showing extreme energy values.
        """

        if self.fc2 is None:
            raise ValueError("FC2 is required for HarmonicReal.")

        self._mesh_dict = self._solve_eigen_equation()
        self._disps = self._get_distribution(t=t, n_samples=n_samples)
        self._supercells = get_structures_from_displacements(
            self._disps, self.supercell
        )
        if self._verbose:
            print("Computing energies and forces using MLP.")
        energies, self._forces = self.eval(self._supercells)
        self._energies_full = energies - self._e0

        if self._verbose:
            print("Eliminating outliers.")
        self._eliminate_outliers()

        if self._verbose:
            print("Computing harmonic potentials and forces.")
        self._energies_harm, self._average_forces = self._compute_harmonic_properties()
        # self._average_forces = self._calc_average_forces()
        return self

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

    @property
    def displacements(self) -> np.ndarray:
        """Return displacements, shape=(n_samples, 3, n_atom)."""
        return np.array(self._disps)

    @property
    def supercells(self) -> list[PolymlpStructure]:
        """Return supercells."""
        return self._supercells

    @property
    def forces(self) -> np.ndarray:
        """Return forces, shape=(n_samples, 3, n_atom)."""
        return np.array(self._forces)

    @property
    def full_potentials(self) -> np.ndarray:
        """Return full potentials, shape=(n_samples)."""
        return self._energies_full

    @property
    def average_full_potential(self) -> float:
        """Return average full potential."""
        return np.average(self._energies_full)

    @property
    def harmonic_potentials(self) -> np.ndarray:
        """Return harmonic potentials, shape=(n_samples)."""
        return self._energies_harm

    @property
    def average_harmonic_potential(self) -> float:
        """Return average harmonic potential."""
        return np.average(self._energies_harm)

    @property
    def anharmonic_potentials(self) -> np.ndarray:
        """Return anharmonic potentials, shape=(n_samples)."""
        return self._energies_full - self._energies_harm

    @property
    def average_anharmonic_potential(self) -> float:
        """Return average anharmonic potentials."""
        return np.average(self.anharmonic_potentials)

    @property
    def static_potential(self) -> float:
        """Return static potential of given supercell."""
        return self._e0

    @property
    def static_forces(self) -> float:
        """Return static forces of given supercell."""
        return self._f0

    @property
    def average_forces(self) -> np.ndarray:
        """Return static forces of given supercell."""
        return self._average_forces

    @property
    def frequencies(self):
        """Return phonon frequencies calculated from effective harmonic H."""
        return self._mesh_dict["frequencies"]

    def _compute_properties(self, t: float = 1000):
        """Compute properties.

        Deprecated.
        """
        freq = self._hide_imaginary_modes(self._mesh_dict["frequencies"])
        nonzero = np.isclose(freq, 0.0) == False

        beta = 1.0 / (Kb * t)
        const_exp = 0.5 * beta * const_planck
        beta_h_freq = const_exp * freq[nonzero]

        occ = np.zeros(freq.shape)
        occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

        e_mode = const_planck * freq
        e_total = np.sum(e_mode * occ) / self.supercell.n_unitcells
        e_total_kJmol = e_total * Avogadro / 1000

        # Calculate free energy
        f_total = np.sum(np.log(2 * np.sinh(beta_h_freq))) / beta
        f_total /= self.supercell.n_unitcells
        f_total_kJmol = f_total * Avogadro / 1000

        self._tp_dict["energy"] = e_total_kJmol
        self._tp_dict["free_energy"] = f_total_kJmol
        return self._tp_dict


#    def _harmonic_residual_forces(self, disps: np.ndarray):
#        """Calculate harmonic forces of a supercell.
#
#        Ideally, these forces must be zero.
#
#        Parameters
#        ----------
#        disps: Displacements. shape=(N3) or (N3, n_samples)
#
#        Return
#        ------
#        forces: Harmonic forces. shape=(n_samples, 3, n_atoms)
#        """
#        N3 = self.fc2.shape[0] * self.fc2.shape[2]
#        fc2 = np.transpose(self.fc2, (0, 2, 1, 3))
#        fc2 = np.reshape(fc2, (N3, N3))
#        forces = -(fc2 @ disps).reshape((self.fc2.shape[0], 3, -1))
#        forces = forces.transpose((1, 0, 2))
#        if forces.shape[2] == 1:
#            return forces[:, :, 0]
#        return forces.transpose((2, 0, 1))
#
#    def _calc_average_forces(self):
#        """Compute average forces."""
#        residual_f = (
#            self._harmonic_residual_forces(
#                self._disps.transpose((2, 1, 0)).reshape((-1, self._disps.shape[0]))
#            )
#            + self._f0
#        )
#        self._average_forces = np.mean(self._forces - residual_f, axis=0)
#        return self._average_forces
