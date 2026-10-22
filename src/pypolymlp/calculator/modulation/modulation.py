"""Class for modulation."""

import itertools
from typing import Optional

import numpy as np
from phonopy import Phonopy

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.phonon_utils import produce_harmonic_force_constants
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)

from .modulation_utils import optimize_geometry


class PolymlpModulation:
    """Class for calculating modulated structures."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        properties: Properties,
        pressure: float = 0.0,
        gtol: float = 1e-4,
        maxiter: int = 10000,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure format
        supercell_matrix: Supercell matrix for computing phonon.
        properties: Properties instance.
        """
        self._prop = properties
        self._pressure = pressure
        self._verbose = verbose

        self._spacegroup = None
        self.optimize_geometry(unitcell, gtol=gtol, maxiter=maxiter)
        self._supercell_matrix = supercell_matrix

        self._structures_children = None
        self._ph = None

    def optimize_geometry(
        self,
        st: PolymlpStructure,
        gtol: float = 1e-4,
        maxiter: int = 10000,
    ):
        """Optimize geometry."""
        self._enthalpy, self._unitcell, self._spacegroup = optimize_geometry(
            st,
            self._prop,
            gtol=gtol,
            maxiter=maxiter,
            pressure=self._pressure,
            verbose=self._verbose,
        )
        return (self._enthalpy, self._unitcell, self._spacegroup)

    def set_qpoints(self, ph: Phonopy, max_denom: int = 6):
        """Enumerate q-point vectors."""
        symops = ph.symmetry.symmetry_operations
        rotations = symops["rotations"]

        cand = []
        seen = set()
        for denom in range(1, max_denom + 1):
            for numer in range(-denom, denom):
                ratio = np.round(numer / denom, 6)
                if abs(ratio) <= 0.5 and ratio not in seen:
                    seen.add(ratio)
                    cand.append((numer, denom))

        uniq_qpoints = {}
        for prod in itertools.product(cand, repeat=3):
            qpoint = np.array([num / denom for num, denom in prod])
            if np.isclose(np.linalg.norm(qpoint), 0.0):
                continue

            rotated = np.array([rot @ qpoint for rot in rotations])
            idx = np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))
            representative = tuple(np.round(rotated[idx[-1]], 6))
            uniq_qpoints[representative] = prod

        self._uniq_qpoints = np.array(list(uniq_qpoints.values()))
        return self._uniq_qpoints

    def run_phonon_and_modulations(
        self,
        max_denom: int = 3,
        amplitude: float = 0.1,
        phase: float = 0.0,
        fc2: Optional[np.ndarray] = None,
    ):
        """Run modulated structures."""
        self._ph = Phonopy(
            structure_to_phonopy_cell(self._unitcell),
            self._supercell_matrix,
            primitive_matrix="P",
        )
        if fc2 is None:
            fc2 = produce_harmonic_force_constants(
                self._ph,
                self._prop,
                displacements=0.01,
            )
        self._ph.force_constants = fc2

        uniq_qpoints = self.set_qpoints(self._ph, max_denom=max_denom)
        self._structures_children = []
        for prod in uniq_qpoints:
            q_point = np.array([p[0] / p[1] for p in prod])
            self._ph.run_qpoints(q_point)
            results = self._ph.get_qpoints_dict()
            band_indices = np.where(results["frequencies"] < 0)[0]

            if len(band_indices) == 0:
                continue

            dimension = [p[1] for p in prod]
            phonon_modes = [[q_point, b, amplitude, phase] for b in band_indices]
            self._ph.run_modulations(dimension, phonon_modes)
            mods = self._ph.get_modulated_supercells()

            for m in mods:
                self._structures_children.append(phonopy_cell_to_structure(m))

        return self._structures_children

    @property
    def unitcell(self):
        """Return parent modulation."""
        return self._unitcell

    @property
    def enthalpy(self):
        """Return enthalpy."""
        return self._enthalpy

    @property
    def spacegroup(self):
        """Return spacegroup."""
        return self._spacegroup

    @property
    def structure_children(self):
        """Return structures of children."""
        return self._structures_children

    @property
    def is_imaginary(self):
        """Return whether imaginary frequencies are found."""
        return len(self._structures_children) > 0
