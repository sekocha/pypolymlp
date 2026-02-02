"""Dataclass of SSCHA results."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PolymlpDataSSCHA:
    """Dataclass of sscha results.

    Parameters
    ----------
    temperature: Temperature (K).
    static_potential: Potential energy of equilibrium structure at 0 K.
    harmonic_potential: Harmonic potential energy for effective FC2.
    harmonic_free_energy: Harmonic free energy for effective FC2.
    average_potential: Averaged full potential energy.
    anharmonic_free_energy: Anharmonic free energy,
                            average_potential - harmonic_potential.
    free_energy: Free energy (harmonic_free_energy + anharmonic_free_energy).
                 Static potential of initial structure is not included.
    entropy: Entropy for effective FC2.
    harmonic_heat_capacity: Harmonic heat capacity for effective FC2.
    delta: Difference between old FC2 and updated FC2.
    converge: SSCHA calculations are converged or not.
    imaginary: Imaginary frequencies are found or not.
    """

    temperature: float
    static_potential: float
    harmonic_potential: float
    harmonic_free_energy: float
    average_potential: float
    anharmonic_free_energy: float
    free_energy: Optional[float] = None

    entropy: Optional[float] = None
    harmonic_heat_capacity: Optional[float] = None
    static_forces: Optional[bool] = None
    average_forces: Optional[bool] = None

    delta: Optional[float] = None
    converge: Optional[bool] = None
    imaginary: Optional[bool] = None

    def __post_init__(self):
        """Post init method."""
        self.free_energy = self.harmonic_free_energy + self.anharmonic_free_energy
