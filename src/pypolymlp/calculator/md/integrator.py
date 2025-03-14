"""Class for integrators."""

from typing import Optional

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.calculator.md.hamiltonian_base import HamiltonianBase
from pypolymlp.calculator.md.integrator_base import IntegratorBase
from pypolymlp.calculator.md.utils import initialize_velocity
from pypolymlp.core.data_format import PolymlpStructure

#
# import numpy as np


class VelocityVerlet(IntegratorBase):
    """Class for velocity Verlet integrator.

    p(dt/2) = p(0) + (dt/2) * F(0)
    x(dt) = x(0) + (dt/m) * p(dt/2)
    Force calculation for x(dt)
    p(dt) = p(dt/2) + (dt/2) * F(dt)
    """

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
        initialize_velocity(self.structure, self.md_params)
        # self.v = ***
        # self.p = ***
        self.eval()

    def run(self, delta_t: float = 2.0, n_steps: int = 10000):
        """Run velocity Verlet integrator.

        Algorithm
        ---------
        1. p(dt/2) = p(0) + (dt/2) * F(0)
        2. x(dt) = x(0) + (dt/m) * p(dt/2)
        3. Force calculation for x(dt)
        4. p(dt) = p(dt/2) + (dt/2) * F(dt)
        """
        # if self.p is None:
        #     raise RuntimeError("Momentum not found.")

        # TODO: Consider unit.
        self._dtm = delta_t / self.masses
        print(self._dtm)

        for i in range(n_steps):
            self._run_single_iteration(delta_t=delta_t)

    def _run_single_iteration(self, delta_t: float = 2.0):
        """Run single iteration of integrator."""
        self.p = self.p + (0.5 * delta_t) * self.f
        self.x = self.x + self._dtm * self.p
        self.eval()
        self.p = self.p + (0.5 * delta_t) * self.f
        return self
