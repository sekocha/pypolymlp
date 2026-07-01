"""Class for calculating properties using hybrid polymlp."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.properties_single import PropertiesSingle
from pypolymlp.calculator.utils.properties_base import PropertiesBase
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.params import PolymlpParams


class PropertiesHybrid(PropertiesBase):
    """Class for calculating properties using a hybrid polymlp model."""

    def __init__(
        self,
        pot: Optional[list[str]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[list[np.ndarray]] = None,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        """
        super().__init__()
        if pot is not None:
            if not isinstance(pot, list):
                raise ValueError("Parameters in PropertiesHybrid must be lists.")
            self._props = [PropertiesSingle(pot=p) for p in pot]
        else:
            self._props = [
                PropertiesSingle(params=p, coeffs=c) for p, c in zip(params, coeffs)
            ]

    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure."""
        energy, force, stress = self._props[0].eval(st, use_openmp=use_openmp)
        for prop in self._props[1:]:
            e_single, f_single, s_single = prop.eval(st, use_openmp=use_openmp)
            energy += e_single
            force += f_single
            stress += s_single
        return energy, force, stress

    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        energies, forces, stresses = self._props[0].eval_multiple(structures)
        for prop in self._props[1:]:
            e_single, f_single, s_single = prop.eval_multiple(structures)
            energies += e_single
            for i, f1 in enumerate(f_single):
                forces[i] += f1
            stresses += s_single
        return energies, forces, stresses

    @property
    def elements(self):
        """Return elements."""
        return self._props[0].elements

    @property
    def params(self):
        """Return parameters for hybrid model."""
        return PolymlpParams([prop.params for prop in self._props])

    @property
    def pot(self):
        """Return potential filename."""
        return None

    def save(self, verbose: bool = False):
        """Save properties to files."""
        return None
