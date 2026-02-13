"""Utility functions for MD."""

import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import Kb, M_StoAng_Fs, MasstoKG


def initialize_velocity(structure: PolymlpStructure, md_params: MDParams):
    """Generate initial velocity.

    Return
    ------
    velocities: Velocities of atoms in unit of angstroms/fs.

    Velocities are randomly sampled from normal distribution with sigma = sqrt(kT/m).
    """
    if md_params.ensemble == "nvt":
        sigma_const = (Kb * md_params.temperature) / MasstoKG
        sigma_atom = np.sqrt(sigma_const * np.reciprocal(structure.masses))
        vel = [np.random.normal(loc=0.0, scale=sigma, size=3) for sigma in sigma_atom]
    elif md_params.ensemble == "nve":
        pass

    vel = (np.array(vel) * M_StoAng_Fs).T
    return vel
