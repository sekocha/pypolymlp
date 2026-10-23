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

from .modulation_utils import ModulationNode, optimize_geometry


class PolymlpModulation:
    """Class for calculating modulated structures."""

    def __init__(
        self,
        properties: Properties,
        pressure: float = 0.0,
        verbose: bool = False,
        verbose_go: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        properties: Properties instance.
        """
        self._prop = properties
        self._pressure = pressure
        self._verbose = verbose
        self._verbose_go = verbose_go

    def optimize_geometry(
        self,
        st: PolymlpStructure,
        gtol: float = 1e-4,
        maxiter: int = 10000,
    ):
        """Optimize geometry."""
        enthalpy, unitcell, spacegroup = optimize_geometry(
            st,
            self._prop,
            gtol=gtol,
            maxiter=maxiter,
            pressure=self._pressure,
            verbose=self._verbose_go,
        )
        if enthalpy is None:
            return None

        node = ModulationNode(unitcell, enthalpy, spacegroup)
        return node

    def optimize_geometry_multiple(
        self,
        structures: list,
        gtol: float = 1e-4,
        maxiter: int = 10000,
        max_n_atom: int = 48,
    ):
        """Optimize geometry of multiple structures."""
        children = []
        for st in structures:
            if len(st.elements) > max_n_atom:
                continue
            try:
                # Structures are refined using with_sym=True
                node_child = self.optimize_geometry(st, gtol=gtol, maxiter=maxiter)
                if node_child is None:
                    if self._verbose:
                        print("Geometry optimization not succeed.", flush=True)
                else:
                    children.append(node_child)
            except RuntimeError:
                if self._verbose:
                    print("Geometry optimization failed.", flush=True)
                continue
        return children

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
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
        gtol: float = 1e-4,
        maxiter: int = 10000,
        max_denom: int = 3,
        amplitude: float = 0.1,
        phase: float = 0.0,
        fc2: Optional[np.ndarray] = None,
    ):
        """Run modulated structures.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure format
        supercell_matrix: Supercell matrix for computing phonon.
        """
        node = self.optimize_geometry(unitcell, gtol=gtol, maxiter=maxiter)
        ph = Phonopy(structure_to_phonopy_cell(unitcell), supercell_matrix)
        # primitive_matrix="P",
        if fc2 is None:
            fc2 = produce_harmonic_force_constants(ph, self._prop, displacements=0.01)
        ph.force_constants = fc2

        uniq_qpoints = self.set_qpoints(ph, max_denom=max_denom)
        structures_children = []
        for prod in uniq_qpoints:
            q_point = np.array([p[0] / p[1] for p in prod])
            ph.run_qpoints(q_point)
            results = ph.get_qpoints_dict()
            band_indices = np.where(results["frequencies"] < -0.1)[0]

            if len(band_indices) == 0:
                continue

            dimension = [p[1] for p in prod]
            phonon_modes = [[q_point, b, amplitude, phase] for b in band_indices]
            ph.run_modulations(dimension, phonon_modes)
            mods = ph.get_modulated_supercells()

            for m in mods:
                structures_children.append(phonopy_cell_to_structure(m))

        node.is_imaginary = len(structures_children) > 0
        node.structures_children = structures_children
        return node
