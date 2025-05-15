"""Class for Verlet integrators."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.calculator.md.hamiltonian_base import HamiltonianBase
from pypolymlp.calculator.md.integrators.integrator_base import IntegratorBase
from pypolymlp.calculator.md.integrators.operator import (
    translate_momenta,
    translate_positions,
)
from pypolymlp.calculator.md.utils import initialize_velocity
from pypolymlp.core.data_format import PolymlpStructure


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
            print(self.e, self.ke, self.e + self.ke)

    def _run_single_iteration(self, delta_t: float = 2.0):
        """Run single iteration of integrator."""
        self.p = translate_momenta(self.p, self.f, 0.5 * delta_t)
        self.x = translate_positions(self.x, self.p, self._dtm)
        self.eval()
        self.p = translate_momenta(self.p, self.f, 0.5 * delta_t)
        return self
