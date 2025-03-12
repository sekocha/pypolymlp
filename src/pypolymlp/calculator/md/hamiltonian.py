"""Class for defining Hamiltonian from polymlp."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.md.hamiltonian_base import HamiltonianBase
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure


class Hamiltonian(HamiltonianBase):
    """Class for defining Hamiltonian from polymlp."""

    def __init__(
        self,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        # super().__init__(polymlp_dev_data, verbose=verbose)
        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._verbose = verbose
        self._energy = None
        self._forces = None
        self._stress_tensor = None

    def eval(self, structure: PolymlpStructure, args: Optional[dict] = None):
        """Evaluate energy, forces, and stress tensor for a single structure.

        Parameters
        ----------
        structure: Structure in PolymlpStructure format.
        args: Parameters used to evaluate properties.

        Return
        ------
        energy: unit: eV/supercell.
        forces: unit: eV/angstrom (3, n_atom)
        stress: unit: eV/supercell (6) in the order of xx, yy, zz, xy, yz, zx.

        Variable args is not used in this function.
        """
        self._energy, self._forces, self._stress_tensor = self.prop.eval(structure)
        return self._energy, self._forces, self._stress_tensor

    @property
    def energy(self):
        """Return energy.

        Return
        ------
        energy: unit: eV/supercell.
        """
        return self._energy

    @property
    def forces(self):
        """Return forces.

        Return
        ------
        forces: unit: eV/angstrom (3, n_atom)
        """
        return self._forces

    @property
    def stress_tensor(self):
        """Return stress tensor.

        Return
        ------
        stress: unit: eV/supercell (6) in the order of xx, yy, zz, xy, yz, zx.
        """
        return self._stress_tensor


class HamiltonianFC2(HamiltonianBase):
    """Class for defining Hamiltonian from second-order force constants."""

    def __init__(self, fc2: np.ndarray, verbose: bool = False):
        """Init method.

        Parameters
        ----------
        fc2: Second-order force constants. shape = (N, N, 3, 3).
        """

        self._natom = fc2.shape[0]
        N3 = self._natom * 3
        self._fc2 = fc2.transpose((0, 2, 1, 3)).reshape((N3, N3))
        self._verbose = verbose
        self._energy = None
        self._forces = None
        self._stress_tensor = None

    def eval(self, structure: PolymlpStructure, args: dict):
        """Evaluate energy, forces, and stress tensor for a single structure.

        Parameters
        ----------
        structure: Structure in PolymlpStructure format.
        args: Parameters used to evaluate properties.

        Return
        ------
        energy: unit: eV/supercell.
        forces: unit: eV/angstrom (3, n_atom)
        stress: unit: eV/supercell (6) in the order of xx, yy, zz, xy, yz, zx.

        Key `equilibrium_structure` is required in args.
        """

        if "equilibrium_structure" not in args:
            raise RuntimeError("Key equilibrium_structure is required.")

        # 1. get displacements from structure.
        disp = None
        # 2. Calculate properties.
        self._forces = -self._fc2 @ disp
        self._energy = disp @ self._fc2 @ disp

        self._forces = self._forces.reshape((self._natom, 3)).T
        # self._energy, self._forces, self._stress_tensor = self.prop.eval(structure)

    @property
    def energy(self):
        """Return energy."""
        return self._energy

    @property
    def forces(self):
        """Return forces."""
        return self._forces

    @property
    def stress_tensor(self):
        """Return stress tensor."""
        return self._stress_tensor
