"""Class for computing elastic constants."""

import copy

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.data_format import PolymlpStructure

# from pypolymlp.utils.tensor_utils import compute_spg_projector_O4


class PolymlpElastic:
    """Class for computing elastic constants."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        properties: Properties,
        geometry_optimization: bool = True,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure.
        properties: Properties instance.
        """
        self._prop = properties
        self._unitcell = unitcell
        self._verbose = verbose

        self._voidt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        self._to_voidt_order = [0, 1, 2, 4, 5, 3]
        self._elastic_constants = None

        if geometry_optimization:
            geometry = GeometryOptimization(
                unitcell,
                properties,
                relax_cell=True,
                relax_volume=True,
                relax_positions=True,
                with_sym=True,
                pressure=0.0,
                verbose=verbose,
            ).run(gtol=1e-3)
            self._unitcell = geometry.structure

        _, _, stress = self._prop.eval(self._unitcell)
        self._eq_stress = stress[self._to_voidt_order]

        # self._proj4 = compute_spg_projector_O4(self._unitcell)

    def eval(self, structures: list[PolymlpStructure]):
        """Evaluate stress tensors.

        Return
        ------
        Stress tensor in the order of Voigt notation.
        (1: xx, 2: yy, 3: zz, 4: yz, 5: zx, 6: xy)
        """
        _, _, stresses = self._prop.eval_multiple(structures)
        stresses = stresses[:, self._to_voidt_order]
        stresses -= self._eq_stress
        stresses = convert_stresses_in_gpa(stresses, structures)
        return stresses

    def run(self, n_samples: int = 7, eps: float = 1e-4):
        """Run elastic constant calculation."""
        magnitudes = np.linspace(-eps, eps, n_samples)
        magnitudes = magnitudes[np.logical_not(np.isclose(magnitudes, 0.0))]

        elastic_consts = np.zeros((6, 6))
        for voidt1, (i, j) in enumerate(self._voidt):
            structures = []
            for mag in magnitudes:
                st = copy.deepcopy(self._unitcell)
                cauchy_deformation = np.eye(3)
                cauchy_deformation[i, j] += mag
                cauchy_deformation[j, i] += mag
                st.axis = cauchy_deformation @ st.axis
                structures.append(st)

            stresses = self.eval(structures)
            X = magnitudes * 2.0
            X = X[:, np.newaxis]
            for voidt2 in range(6):
                y = -stresses[:, voidt2]
                slope, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                elastic_consts[voidt1, voidt2] = slope[0]
                elastic_consts[voidt2, voidt1] = slope[0]
        self._elastic_constants = elastic_consts
        return self

    def write_elastic_constants(self, filename: str = "polymlp_elastic.yaml"):
        """Save elastic constants."""
        with open(filename, "w") as f:
            print("elastic_constants:", file=f)
            print("  unit: GPa", file=f)

            ids = [
                (1, 1),
                (2, 2),
                (3, 3),
                (1, 2),
                (1, 3),
                (2, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (1, 4),
                (1, 5),
                (1, 6),
                (2, 4),
                (2, 5),
                (2, 6),
                (3, 4),
                (3, 5),
                (3, 6),
                (4, 5),
                (4, 6),
                (5, 6),
            ]
            for i, j in ids:
                prefix = "  c_" + str(i) + str(j) + ":"
                if self._elastic_constants[i - 1][j - 1] > 1e-8:
                    str1 = "{:.3f}".format(self._elastic_constants[i - 1][j - 1])
                    print(prefix, str1, file=f)
                else:
                    print(prefix, "0", file=f)
        return self

    @property
    def elastic_constants(self):
        """Return elastic constants.

        Return
        ------
        elastic_constants: Elastic constants in Voigt notation.
                           (1: xx, 2: yy, 3: zz, 4: yz, 5: zx, 6: xy)
        """
        return self._elastic_constants
