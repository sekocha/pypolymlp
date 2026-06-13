"""Class for computing elastic constants."""

import copy
import io

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.properties_utils import convert_stresses_in_gpa
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoGPa
from pypolymlp.utils.tensor_utils_O4 import compute_projector_O4


def write_elastic_constants(
    elastic_constants: np.ndarray,
    file: io.IOBase,
    tag: str = "elastic_constants",
):
    """Save elastic constants to a file."""
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

    print(tag + ":", file=file)
    for i, j in ids:
        prefix = "  c_" + str(i) + str(j) + ":"
        str1 = "{:.3f}".format(elastic_constants[i - 1][j - 1])
        print(prefix, str1, file=file)
        # if elastic_constants[i - 1][j - 1] > 1e-8:
        #     str1 = "{:.3f}".format(elastic_constants[i - 1][j - 1])
        #     print(prefix, str1, file=file)
        # else:
        #     print(prefix, "0", file=file)
    return file


class PolymlpElastic:
    """Class for computing elastic constants."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        properties: Properties,
        geometry_optimization: bool = True,
        gtol: float = 1e-3,
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
        self._geometry = None

        self._adiabatic_correction = None
        self._temperature = None
        if not isinstance(self._prop, Properties):
            self._temperature = self._prop.temperature

        if geometry_optimization:
            self._geometry = GeometryOptimization(
                unitcell,
                properties,
                relax_cell=True,
                relax_volume=True,
                relax_positions=True,
                with_sym=True,
                pressure=0.0,
                verbose=verbose,
            ).run(gtol=gtol)
            self._unitcell = self._geometry.structure

        _, _, stress = self._prop.eval(self._unitcell)
        self._eq_stress = stress[self._to_voidt_order]

        self._sym_proj = compute_projector_O4(self._unitcell)

    def eval(self, structures: list[PolymlpStructure]):
        """Evaluate stress tensors.

        Return
        ------
        stresses: Array of stress tensors (GPa) in the order of Voigt notation.
                (1: xx, 2: yy, 3: zz, 4: yz, 5: zx, 6: xy)
        """
        _, _, stresses = self._prop.eval_multiple(structures)
        stresses = stresses[:, self._to_voidt_order]
        stresses -= self._eq_stress
        stresses = convert_stresses_in_gpa(stresses, structures)
        return stresses

    def eval_adiabatic(self):
        """Evaluate stress tensors and entropy for unitcell.

        Return
        ------
        stress: Stress tensor (GPa) in the order of Voigt notation.
                (1: xx, 2: yy, 3: zz, 4: yz, 5: zx, 6: xy)
        entropy: Entropy value in eV/K/unitcell.
        """
        _, _, stress = self._prop.eval(self._unitcell)
        stress = stress[self._to_voidt_order]
        stress -= self._eq_stress
        stress = convert_stresses_in_gpa([stress], [self._unitcell])[0]

        entropy = self._prop.entropy
        return stress, entropy

    def _symmetrize(self, elastic_constants: np.ndarray):
        """Symmetrize elastic constants."""
        el_full = np.zeros((3, 3, 3, 3))
        for voidt1, (i, j) in enumerate(self._voidt):
            for voidt2, (k, l) in enumerate(self._voidt):
                el_full[i, j, k, l] = elastic_constants[voidt1, voidt2]
                el_full[i, j, l, k] = elastic_constants[voidt1, voidt2]
                el_full[j, i, k, l] = elastic_constants[voidt1, voidt2]
                el_full[j, i, l, k] = elastic_constants[voidt1, voidt2]
        el_full = (self._sym_proj @ el_full.reshape(-1)).reshape((3, 3, 3, 3))

        for voidt1, (i, j) in enumerate(self._voidt):
            for voidt2, (k, l) in enumerate(self._voidt):
                elastic_constants[voidt1, voidt2] = el_full[i, j, k, l]
        return elastic_constants

    def run(self, n_samples: int = 7, eps: float = 1e-4):
        """Run elastic constant calculation."""
        magnitudes = np.linspace(-eps, eps, n_samples)
        magnitudes = magnitudes[np.logical_not(np.isclose(magnitudes, 0.0))]

        elastic_consts = np.zeros((6, 6))
        for voidt1, (i, j) in enumerate(self._voidt):
            if self._verbose:
                print("- Strain", (i, j), flush=True)
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

        self._elastic_constants = self._symmetrize(elastic_consts)
        return self

    def run_adiabatic(self, n_samples: int = 7, eps: float = 60.0):
        """Run adiabatic contribution."""
        if isinstance(self._prop, Properties):
            raise RuntimeError("Adiabatic calculation requires SSCHA properties.")
        if self._verbose:
            print("Calculating adiabatic correction", flush=True)

        temperatures = np.linspace(-eps, eps, n_samples) + self._temperature
        stress_all, entropy_all = [], []
        for temp in temperatures:
            if self._verbose:
                print("- temperature", temp, flush=True)
            self._prop.temperature = temp
            stress, entropy = self.eval_adiabatic()
            entropy = entropy * EVtoGPa / self._unitcell.volume
            stress_all.append(stress)
            entropy_all.append(entropy)

        stress_all = np.array(stress_all)
        entropy_all = np.array(entropy_all)

        stress_deriv = np.zeros(6)
        for voidt1 in range(6):
            slope, intercept = np.polyfit(temperatures, stress_all[:, voidt1], 1)
            stress_deriv[voidt1] = slope
            if self._verbose:
                pred = slope * temperatures + intercept
                true = stress_all[:, voidt1]
                print("Prediction of dstress/dT for Voidt", voidt1, flush=True)
                for temp, val1, val2 in zip(temperatures, true, pred):
                    print("", temp, np.round(val1, 5), np.round(val2, 5), flush=True)

        entropy_deriv, intercept = np.polyfit(temperatures, entropy_all, 1)
        if self._verbose:
            pred = entropy_deriv * temperatures + intercept
            print("Prediction of dS/dT", flush=True)
            for temp, val1, val2 in zip(temperatures, entropy_all, pred):
                print("", temp, np.round(val1, 5), np.round(val2, 5), flush=True)

        self._adiabatic_correction = (
            np.outer(stress_deriv, stress_deriv) / entropy_deriv
        )
        return self

    def write_elastic_constants(self, filename: str = "polymlp_elastic.yaml"):
        """Save elastic constants."""
        if self._elastic_constants is None:
            return None

        with open(filename, "w") as f:
            if self._temperature is not None:
                print("temperature:", self._temperature, file=f)
                print(file=f)

            print("unit: GPa", file=f)
            print(file=f)

            write_elastic_constants(
                self._elastic_constants,
                file=f,
                tag="elastic_constants",
            )
            if self._adiabatic_correction is not None:
                print(file=f)
                write_elastic_constants(
                    self._elastic_constants + self._adiabatic_correction,
                    file=f,
                    tag="elastic_constants_adiabatic",
                )
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

    @property
    def elastic_constants_adiabatic(self):
        """Return adabatic elastic constants.

        Return
        ------
        elastic_constants: Adiabatic elastic constants in Voigt notation.
                           (1: xx, 2: yy, 3: zz, 4: yz, 5: zx, 6: xy)
        """
        if self._elastic_constants is None:
            return None
        if self._adiabatic_correction is None:
            return None
        return self._elastic_constants + self._adiabatic_correction

    @property
    def unitcell(self):
        """Return unit cell.

        If geometry optimization is performed, equilibrium unit cell is returned.

        Return
        ------
        unitcell: unitcell
        """
        return self._unitcell
