#!/usr/bin/env python

from typing import Optional

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from pypolymlp.calculator.properties import Properties

from ase_utils import convert_ase_atoms_to_polymlp_structure

ALL_CHANGES = tuple(all_changes)


class PolymlpASECalculator(Calculator):
    """ASE calculator using pypolymlp."""
    implemented_properties = ("energy", "forces", "stress")

    def __init__(self, potentials: list = None, **kwargs):
        """Initialize PolymlpASECalculator."""
        super().__init__(**kwargs)
        self._properties = None
        if potentials:
            self.set_calculator(potentials)

    def set_calculator(self, potentials: list):
        """Set pypolymlp potential files."""
        self._properties = Properties(pot=potentials)

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: tuple = ("energy", "forces"),
        system_changes: tuple = ALL_CHANGES,
    ):
        """Calculate energy, force, and stress using `pypolymlp`."""
        super().calculate(atoms, properties, system_changes)
        structure = convert_ase_atoms_to_polymlp_structure(atoms)
        energy, forces, stress = self._properties.eval(structure)
        self.results["energy"] = energy
        self.results["forces"] = forces.T
        self.results["stress"] = stress
