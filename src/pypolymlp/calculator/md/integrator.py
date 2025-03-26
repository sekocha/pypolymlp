"""Class for integrators."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.calculator.md.hamiltonian_base import HamiltonianBase
from pypolymlp.calculator.md.integrator_base import IntegratorBase
from pypolymlp.calculator.md.utils import initialize_velocity
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import Avogadro, EVtoJ


class VelocityVerlet(IntegratorBase):
    """Class for velocity Verlet integrator."""

    def __init__(
        self,
        structure: PolymlpStructure,
        md_params: MDParams,
        hamiltonian: HamiltonianBase,
        hamiltonian_args: Optional[dict] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        structure: Initial structure. Masses must be included.
        md_params: Parameters in MD simulation.
        """
        super().__init__(
            structure,
            md_params,
            hamiltonian,
            hamiltonian_args=hamiltonian_args,
            verbose=verbose,
        )
        self._const_ftop = (Avogadro * 1e3) * EVtoJ * 1e-10

    def initialize(self):
        """Initialize simulation."""
        velocities = initialize_velocity(self.structure, self.md_params)
        self.p = velocities * self.masses
        self.eval()
        return self

    def run(self, delta_t: float = 2.0, n_steps: int = 10000):
        """Run velocity Verlet integrator.

        Algorithm
        ---------
        1. p(dt/2) = p(0) + (dt/2) * F(0)
        2. x(dt) = x(0) + (dt/m) * p(dt/2)
        3. Force calculation for x(dt)
        4. p(dt) = p(dt/2) + (dt/2) * F(dt)
        """
        if self.p is None:
            raise RuntimeError("Momentum not found.")

        self._dtm = delta_t * np.reciprocal(self.masses)
        for i in range(n_steps):
            self._run_single_iteration(delta_t=delta_t)

    def _run_single_iteration(self, delta_t: float = 2.0):
        """Run single iteration of integrator."""
        force = self._const_ftop * self.f
        self.p = self.p + (0.5 * delta_t) * force
        self.x = self.x + self.p * self._dtm
        self.eval()
        force = self._const_ftop * self.f
        self.p = self.p + (0.5 * delta_t) * force
        return self
