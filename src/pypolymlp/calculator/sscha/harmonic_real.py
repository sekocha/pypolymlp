#!/usr/bin/env python 
import numpy as np

from phonopy import Phonopy
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

    def __init__(self, supercell_dict, n_unitcells, fc2):

        self.supercell = supercell_dict
        self.supercell['n_unitcells'] = n_unitcells
        self.fc2 = fc2

        self.mesh_dict = None
        self.tp_dict = None

        self.__set_mass()
        self.__set_inverse_axis()
        self.__check_n_unitcells()

    def __set_mass(self):
        if not 'masses' in self.supercell:
            mass_table = mass_table()
            masses = [mass_table[e] for e in self.supercell['elements']]
            self.supercell['masses'] = masses

    def __set_inverse_axis(self):
        if not 'axis_inv' in self.supercell:
            self.supercell['axis_inv'] = np.linalg.inv(self.supercell['axis'])

    def __set_n_unitcells(self):
        if not 'n_unitcells' in self.supercell:
            raise KeyError(' Key n_unitcells is needed in HarmonicReal.')

    def __solve_eigen_equation(self):

        fc1 = self.fc2.transpose((0,2,1,3))
        size = fc1.shape[0] * fc1.shape[1]
        fc1 = np.reshape(fc1, (size, size))

        masses = np.repeat(self.supercell['masses'], 3)
        masses_sqrt = np.reciprocal(np.sqrt(masses))
        dyn = (np.diag(masses_sqrt) @ fc1) @ np.diag(masses_sqrt)
        square_w, eigvecs = np.linalg.eigh(dyn)
        square_w *= const_sq_angfreq_to_sq_freq_thz # in THz

        negative_square_w = square_w < -1e-8
        positive_square_w = square_w > 1e-8
        freq = np.zeros(square_w.shape)
        freq[positive_square_w] = np.sqrt(square_w[positive_square_w])
        freq[negative_square_w] = -np.sqrt(-square_w[negative_square_w])

        self.mesh_dict['frequencies'] = freq
        self.mesh_dict['eigenvectors'] = eigvecs
        return self.mesh_dict

    def __hide_imaginary_modes(self, freq):
        freq_rev = np.array(freq)
        freq_rev[np.where(freq_rev < 0)] = 0.0
        return freq_rev

    def __compute_properties(self, t=1000):

        freq = self.__hide_imaginary_modes(self.mesh_dict['frequencies'])
        nonzero = np.isclose(freq, 0.0) == False

        beta = 1.0 / (const_bortzmann * t)
        const_exp = 0.5 * beta * const_planck
        beta_h_freq = const_exp * freq[nonzero]

        occ = np.zeros(freq.shape)
        occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

        e_mode = const_planck * freq
        e_total = np.sum(e_mode * occ) / self.supercell['n_unitcells']
        e_total_kJmol = e_total * const_avogadro / 1000

        # free energy
        f_total = np.sum(np.log(2 * np.sinh(beta_h_freq))) / beta
        f_total /= self.supercell['n_unitcells']
        f_total_kJmol = f_total * const_avogadro / 1000

        self.tp_dict['energy'] = e_total_kJmol
        self.tp_dict['free_energy'] = f_total_kJmol
        return self.tp_dict

    def get_harmonic_properties(self, t=1000):
        self.mesh_dict = self.__solve_eigen_equation()
        self.tp_dict = self.__compute_properties(t=t)
        return (self.mesh_dict, self.tp_dict)

    def get_distribution_from_mesh_dict(self, t=1000, n_samples=100):

        freq = self.__hide_imaginary_modes(self.mesh_dict['frequencies'])
        nonzero = np.isclose(freq, 0.0) == False

        beta = 1.0 / (const_bortzmann * t)
        const_exp = 0.5 * beta * const_planck
        beta_h_freq = const_exp * freq[nonzero]

        occ = np.zeros(freq.shape)
        occ[nonzero] = 0.5 * np.reciprocal(np.tanh(beta_h_freq))

        eigvecs = self.mesh_dict['eigenvectors']
        ''' arbitrary setting'''
        amplitudes = np.ones(freq.shape) * 0.01

        rec_freq = np.array([1/f for f in freq[nonzero]])
        amplitudes[nonzero] = const_amplitude * occ[nonzero] * rec_freq

        cov = np.diag(amplitudes)
        mean = np.zeros(cov.shape[0])
        disp_normal_coords = np.random.multivariate_normal(mean, 
                                                           cov, 
                                                           n_samples)

        masses_sqrt = np.sqrt(np.repeat(self.supercell['masses'], 3))
        dot1 = eigvecs @ disp_normal_coords.T
        disps = (np.diag(np.reciprocal(masses_sqrt)) @ dot1).T
        return disps

    def get_distribution(self, t=1000, n_samples=100):
        mesh, tp = self.get_harmonic_properties(t=1000)
        disps = self.get_distribution_from_mesh_dict(t=t, n_samples=n_samples)
        return disps
