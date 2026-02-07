"""Base class for integrators."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from pypolymlp.calculator.md.data import MDParams
from pypolymlp.calculator.md.hamiltonian_base import HamiltonianBase
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoJ, M_StoAng_Fs, MasstoKG


class IntegratorBase(ABC):
    """Base class for integrators."""

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
        self._structure = structure
        self._initialize_structure()
        self._md_params = md_params
        self._hamiltonian = hamiltonian
        self._hamiltonian_args = hamiltonian_args
        self._verbose = verbose

        self._e = None
        self._f = None
        self._s = None
        self._v = None
        self._p = None
        self._ke = None
        if self._md_params.save_history:
            self._energies = []
            self._forces = []

    def _initialize_structure(self):
        """Initialize structure."""
        if self.masses is None:
            raise RuntimeError("Masses not found.")
        self._structure.masses = np.array(self._structure.masses)
        self._structure.set_positions_cartesian()

    @abstractmethod
    def run(self):
        """Run integrator."""
        pass

    def eval(self):
        """Evaluate energy, forces, and stress tensor."""
        e, f, s = self._hamiltonian.eval(self._structure, self._hamiltonian_args)
        self._e = e
        self._f = f
        self._s = s
        if self._md_params.save_history:
            self._energies.append(e)
            self._forces.append(f)
        return e, f, s

    @property
    def hamltonian(self):
        """Return Hamiltonian."""
        return self._hamiltonian

    @property
    def structure(self):
        """Return current structure."""
        return self._structure

    @property
    def md_params(self):
        """Return parameters in MD."""
        return self._md_params

    @property
    def x(self):
        """Return current positions in cartesian."""
        return self._structure.positions_cartesian

    @x.setter
    def x(self, x: np.ndarray):
        """Set positions in cartesian."""
        self._structure.positions_cartesian = x
        inv = np.linalg.inv(self._structure.axis)
        self._structure.positions = inv @ x

    @property
    def p(self):
        """Return current momenta in (g/mol)*(angstrom/fs)."""
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
    def e(self):
        """Return current potential energy in eV/cell."""
        return self._e

    @property
    def f(self):
        """Return current force in eV/angstrom."""
        return self._f

    @property
    def ke(self):
        """Return current kinetic energy in eV/cell."""
        if self.p is not None:
            norm = np.linalg.norm(self.p, axis=0) ** 2
            norm /= self.masses
            const = 0.5 * MasstoKG / M_StoAng_Fs**2 / EVtoJ
            self._ke = const * np.sum(norm)
        else:
            raise RuntimeError("No kinetic energy function.")

        return self._ke

    @property
    def masses(self):
        """Return masses."""
        return self._structure.masses

    @masses.setter
    def masses(self, m: np.ndarray):
        """Set masses."""
        self._structure.masses = m

    @property
    def energies(self):
        """Return energies in simulation."""
        return np.array(self._energies)

    @property
    def forces(self):
        """Return forces in simulation."""
        return np.array(self._forces)
