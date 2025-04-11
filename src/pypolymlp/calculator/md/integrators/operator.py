"""Functions for time evolution operators."""

import numpy as np

from pypolymlp.core.units import Avogadro, EVtoJ

# Units
# -----
# r: positions, angstrom.
# v: velocities, angstrom/fs
# p: momentum, (g/mol)*(angstrom/fs)
# f: force, eV/angstrom

# force (eV/angstrom) -> momentum (g/mol)*(angstrom/fs)
const_f_to_p = (Avogadro * 1e3) * EVtoJ * 1e-10


def translate_positions(positions: np.ndarray, momenta: np.ndarray, dtm: np.ndarray):
    r"""Translate positions by standard Hamiltonian term.

    Parameters
    ----------
    positions: Cartesian positions in angstrom.
    momenta: Momenta of atoms in (g/mol)*(angstrom/fs).
    dtm: Time divided by mass.

    exp[(p_i / m_i) (\partial/\partial r_i)] r_i = r_i + (p_i / m_i) * dt
    """
    return positions + momenta * dtm


def translate_momenta(momenta: np.ndarray, forces: np.ndarray, dt: float):
    r"""Translate momenta by standard Hamiltonian term.

    Parameters
    ----------
    momenta: Momenta of atoms in (g/mol)*(angstrom/fs).
    forces: Forces acting on atoms in eV/angstrom.
    dt: Time in fs.

    exp[F_i (\partial/\partial p_i)] p_i = p_i + F_i * dt
    """
    return momenta + (const_f_to_p * dt) * forces
