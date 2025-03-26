"""Dataclass for MD simulations."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MDParams:
    """Dataclass of parameters in MD simulation.

    Parameters
    ----------
    delta_t: Time step in fs.
    n_steps: Number of steps.
    ensemble: Ensemble.
    energy: Total energy in NVE ensemble.
    temperature: Temperature in NVT ensemble.
    """

    delta_t: float = 2.0
    n_steps: int = 10000
    ensemble: Literal["nve", "nvt"] = "nvt"
    energy: Optional[float] = None
    temperature: Optional[float] = None
    save_history: bool = True

    def __post_init__(self):
        """Post-init method."""
        self._error_check()

    def _error_check(self):
        """Check errors."""
        if self.ensemble == "nvt" and self.temperature is None:
            raise RuntimeError("Temperature not found for NVT.")
        if self.ensemble == "nve" and self.energy is None:
            raise RuntimeError("Energy not found for NVE.")
        return self
