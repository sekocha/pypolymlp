"""Base class for integrators."""

from abc import ABC, abstractmethod

import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.core.data_format import PolymlpStructure


class IntegratorBase(ABC):
    """Base class for integrators."""

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
        self._structure = structure
        self._initialize_structure()
        self._md_params = md_params
        self._verbose = verbose

        self._x = None
        self._p = None
        self._force = None

    def _initialize_structure(self):
        """Initialize structure."""
        if self.masses is None:
            raise RuntimeError("Masses not found.")
        self._structure.set_positions_cartesian()

    @abstractmethod
    def run(self):
        """Run integrator."""
        pass

    @property
    def md_params(self):
        """Return parameters in MD."""
        return self._md_params

    @property
    def structure(self):
        """Return current structure."""
        return self._structure

    @property
    def x(self):
        """Return current positions in cartesian."""
        return self._structure.positions_cartesian

    @x.setter
    def x(self, x: np.ndarray):
        """Set positions in cartesian."""
        self._structure.positions_cartesian = x

    @property
    def p(self):
        """Return current momenta."""
        return self._structure.momenta

    @p.setter
    def p(self, p: np.ndarray):
        """Return momenta."""
        self._structure.momenta = p

    @property
    def v(self):
        """Return current velocities."""
        return self._structure.velocities

    @v.setter
    def v(self, v: np.ndarray):
        """Return velocities."""
        self._structure.velocities = v

    @property
    def masses(self):
        """Return masses."""
        return self._structure.masses

    @masses.setter
    def masses(self, m: np.ndarray):
        """Set masses."""
        self._structure.masses = m
