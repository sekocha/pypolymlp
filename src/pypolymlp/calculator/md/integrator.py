"""Class for integrators."""

# from typing import Optional
#
# import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.calculator.md.integrator_base import IntegratorBase
from pypolymlp.core.data_format import PolymlpStructure


class VelocityVerlet(IntegratorBase):
    """Class for velocity Verlet integrator."""

    def __init__(
        self,
        structure: PolymlpStructure,
        md_params: MDParams,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        structure: Initial structure. Masses must be included.
        md_params: Parameters in MD simulation.
        """
        super().__init__(structure, md_params, verbose=verbose)

    def run(self):
        pass
