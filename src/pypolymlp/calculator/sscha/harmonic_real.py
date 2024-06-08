#!/usr/bin/env python
import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.displacements import get_structures_from_displacements
from pypolymlp.core.utils import mass_table

const_avogadro = 6.02214076e23
const_planck = 6.62607015e-22
const_bortzmann = 1.380649e-23

""" const_sq_angfreq_to_sq_freq_thz:
     1.602176634e-19 (eV->J) * 6.02214076e23 (avogadro) * 0.1 / (4 * pi^2)
"""
const_sq_angfreq_to_sq_freq_thz = 2.4440020137144617e2

"""
  const_amplitude:
    J*s/THz -> atomic_mass * angstrom^2
    6.62607015e-34 * 6.02214076e23 * 1e11 / (4 * pi * pi)
"""
const_amplitude = 1.010758017933576


class HarmonicReal:

    def __init__(
        self,
        supercell_dict,
        properties: Properties,
        n_unitcells=None,
        fc2=None,
    ):

        self.supercell = supercell_dict
        self.fc2 = fc2
        if n_unitcells is not None:
            self.supercell["n_unitcells"] = n_unitcells

        self.prop = properties

        self.__mesh_dict = dict()
        self.__tp_dict = dict()

        self.__set_mass()
        self.__set_inverse_axis()
        self.__check_n_unitcells()

        self.__e0 = self.prop.eval(self.supercell)[0]

    def __set_mass(self):
        if "masses" not in self.supercell:
            table = mass_table()
            masses = [table[e] for e in self.supercell["elements"]]
            self.supercell["masses"] = masses

    def __set_inverse_axis(self):
        if "axis_inv" not in self.supercell:
            self.supercell["axis_inv"] = np.linalg.inv(self.supercell["axis"])

    def __check_n_unitcells(self):
        if "n_unitcells" not in self.supercell:
            raise KeyError(" Key n_unitcells is needed in HarmonicReal.")

    def compute_polymlp_properties(self, st_dicts):
        """energies: (n_str)
        forces: (n_str, 3, n_atom)
        """
        energies, forces, _ = self.prop.eval_multiple(st_dicts)
        return np.array(energies), np.array(forces)

    def __solve_eigen_equation(self):

        fc1 = self.fc2.transpose((0, 2, 1, 3))
        size = fc1.shape[0] * fc1.shape[1]
        fc1 = np.reshape(fc1, (size, size))

        masses = np.repeat(self.supercell["masses"], 3)
        masses_sqrt = np.reciprocal(np.sqrt(masses))
        dyn = (np.diag(masses_sqrt) @ fc1) @ np.diag(masses_sqrt)
        square_w, eigvecs = np.linalg.eigh(dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz  # in THz

        negative_square_w = square_w < -1e-8
        positive_square_w = square_w > 1e-8
        freq = np.zeros(square_w.shape)
        freq[positive_square_w] = np.sqrt(square_w[positive_square_w])
        freq[negative_square_w] = -np.sqrt(-square_w[negative_square_w])

        self.__mesh_dict["frequencies"] = freq
        self.__mesh_dict["eigenvectors"] = eigvecs
        return self.__mesh_dict

    def __hide_imaginary_modes(self, freq):
        freq_rev = np.array(freq)
        freq_rev[np.where(freq_rev < 0)] = 0.0
        return freq_rev

    def __compute_properties(self, t=1000):

        freq = self.__hide_imaginary_modes(self.__mesh_dict["frequencies"])
        nonzero = np.isclose(freq, 0.0) == False

        beta = 1.0 / (const_bortzmann * t)
        const_exp = 0.5 * beta * const_planck
        beta_h_freq = const_exp * freq[nonzero]

        occ = np.zeros(freq.shape)
        occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

        e_mode = const_planck * freq
        e_total = np.sum(e_mode * occ) / self.supercell["n_unitcells"]
        e_total_kJmol = e_total * const_avogadro / 1000

        # free energy
        f_total = np.sum(np.log(2 * np.sinh(beta_h_freq))) / beta
        f_total /= self.supercell["n_unitcells"]
        f_total_kJmol = f_total * const_avogadro / 1000

        self.__tp_dict["energy"] = e_total_kJmol
        self.__tp_dict["free_energy"] = f_total_kJmol
        return self.__tp_dict

    def __harmonic_potential(self, disp):

        N3 = self.fc2.shape[0] * self.fc2.shape[2]
        fc1 = np.transpose(self.fc2, (0, 2, 1, 3))
        fc1 = np.reshape(fc1, (N3, N3))
        return 0.5 * disp @ (fc1 @ disp)

    def __get_distribution(self, t=1000, n_samples=100):

        freq = self.__hide_imaginary_modes(self.__mesh_dict["frequencies"])
        nonzero = np.isclose(freq, 0.0) == False

        beta = 1.0 / (const_bortzmann * t)
        const_exp = 0.5 * beta * const_planck
        beta_h_freq = const_exp * freq[nonzero]

        occ = np.zeros(freq.shape)
        occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

        eigvecs = self.__mesh_dict["eigenvectors"]
        """ arbitrary setting"""
        amplitudes = np.ones(freq.shape) * 0.01

        rec_freq = np.array([1 / f for f in freq[nonzero]])
        amplitudes[nonzero] = const_amplitude * occ[nonzero] * rec_freq

        cov = np.diag(amplitudes)
        mean = np.zeros(cov.shape[0])
        disp_normal_coords = np.random.multivariate_normal(mean, cov, n_samples)

        masses_sqrt = np.sqrt(np.repeat(self.supercell["masses"], 3))
        dot1 = eigvecs @ disp_normal_coords.T
        disps = (np.diag(np.reciprocal(masses_sqrt)) @ dot1).T
        disps = disps.reshape((n_samples, -1, 3)).transpose((0, 2, 1))
        return disps

    def __calc_harmonic_potentials(self, t=1000, disps=None):
        """Parameter
        ---------
        disps: None or (n_samples, 3, n_atom)
        """
        if disps is None:
            tp_dict = self.__compute_properties(t=t)
            pot_harmonic = 0.5 * tp_dict["energy"]
        else:
            pot_harmonic = [self.__harmonic_potential(d.T.reshape(-1)) for d in disps]
            pot_harmonic = np.array(pot_harmonic)
        return pot_harmonic

    def __eliminate_outliers(self, log=True, tol_negative=-10):

        energies = np.array(self.__energies_full)
        ids1 = np.where(energies > tol_negative)[0]

        e_ave = np.mean(energies[ids1])
        tol = 2 * abs(e_ave)
        ids2 = np.where(np.abs(energies - e_ave) < tol)[0]
        ids = set(ids1) & set(ids2)

        if log:
            entire_ids = set(list(range(len(energies))))
            outlier_ids = entire_ids - ids
            if len(outlier_ids) > 0:
                print("Outliers are eliminated:")
                print("- pot. energy (average) =", "{:f}".format(e_ave))
                for i in sorted(outlier_ids):
                    print(
                        "- pot. energy (outlier " + str(i),
                        ") =",
                        "{:f}".format(energies[i]),
                    )

        ids = np.array(list(ids))
        self.__disps = self.__disps[ids]
        self.__supercells = [self.__supercells[i] for i in ids]
        self.__forces = self.__forces[ids]
        self.__energies_full = self.__energies_full[ids]
        return self

    def run(self, t=1000, n_samples=100, eliminate_outliers=True):

        if self.fc2 is None:
            raise ValueError("Set FC2 in HarmonicReal.")

        self.__mesh_dict = self.__solve_eigen_equation()
        self.__disps = self.__get_distribution(t=t, n_samples=n_samples)
        self.__supercells = get_structures_from_displacements(
            self.__disps, self.supercell
        )
        energies, self.__forces = self.compute_polymlp_properties(self.__supercells)
        self.__energies_full = energies - self.__e0
        self.__eliminate_outliers()

        self.__energies_harm = self.__calc_harmonic_potentials(t=t, disps=self.__disps)
        return self

    @property
    def force_constants(self):
        """(n_atom, n_atom, 3, 3)"""
        return self.fc2

    @force_constants.setter
    def force_constants(self, fc2):
        """(n_atom, n_atom, 3, 3)"""
        self.fc2 = fc2

    @property
    def displacements(self):
        """(n_samples, 3, n_atom)"""
        return np.array(self.__disps)

    @property
    def supercells(self):
        return self.__supercells

    @property
    def forces(self):
        """(n_samples, 3, n_atom)"""
        return np.array(self.__forces)

    @property
    def full_potentials(self):
        """(n_samples)"""
        return self.__energies_full

    @property
    def average_full_potential(self):
        return np.average(self.__energies_full)

    @property
    def harmonic_potentials(self):
        """(n_samples)"""
        return self.__energies_harm

    @property
    def average_harmonic_potential(self):
        return np.average(self.__energies_harm)

    @property
    def anharmonic_potentials(self):
        return self.__energies_full - self.__energies_harm

    @property
    def average_anharmonic_potential(self):
        return np.average(self.anharmonic_potentials)

    @property
    def static_potential(self):
        return self.__e0

    @property
    def frequencies(self):
        return self.__mesh_dict["frequencies"]
