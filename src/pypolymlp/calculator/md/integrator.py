"""Class for integrators."""

# from typing import Optional
#
# import numpy as np

from pypolymlp.calculator.md.integrator_base import IntegratorBase


class VelocityVerlet(IntegratorBase):
    """Class for velocity Verlet integrator."""

    def __init__(
        self,
        verbose: bool = False,
    ):
        """Init method."""
        super().__init__(verbose=verbose)
