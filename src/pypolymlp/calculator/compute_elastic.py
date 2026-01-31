"""Class for computing elastic constants."""

import copy
from typing import Optional

import numpy as np
import pymatgen as pmg
from pymatgen.analysis.elasticity import DeformedStructureSet, diff_fit

from pypolymlp.calculator.properties import Properties, convert_stresses_in_gpa
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure


class PolymlpElastic:
    """Class for computing elastic constants."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        unitcell_poscar: str,
        pot: Optional[str] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure format.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._unitcell = unitcell
        self._verbose = verbose
        with open(unitcell_poscar) as f:
            self._st_pmg = pmg.core.Structure.from_str(f.read(), fmt="POSCAR")

        self._compute_initial_properties()

    def _get_2d_stress(self, stress: np.ndarray):
        """Convert calculated stress to 2D stress."""
        return np.array(
            [
                [stress[0], stress[3], stress[5]],
                [stress[3], stress[1], stress[4]],
                [stress[5], stress[4], stress[2]],
            ]
        )

    def _compute_initial_properties(self):
        """Compute stress of equilibrium structure.

        stress: xx, yy, zz, xy, yz, zx
        --> p0: xx(1), yy(2), zz(3), yz(4), zx(5), xy(6)
        """
        _, _, stress = self._prop.eval(self._unitcell)
        self._eq_stress = -self._get_2d_stress(stress)
        return self

    def run(self):
        """Run elastic constant calculation."""
        deform = DeformedStructureSet(self._st_pmg)
        strains = [d.green_lagrange_strain for d in deform.deformations]

        structure_deform = []
        for i in range(len(deform)):
            structure = copy.deepcopy(self._unitcell)
            lattice = np.array(deform[i].as_dict()["lattice"]["matrix"])
            structure.axis = lattice.T
            structure_deform.append(structure)

        _, _, stresses = self._prop.eval_multiple(structure_deform)
        stresses = convert_stresses_in_gpa(stresses, structure_deform)
        stresses = -np.array([self._get_2d_stress(s) for s in stresses])

        const = diff_fit(strains, stresses, eq_stress=self._eq_stress)[0]

        ids = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        self._elastic_constants = np.array(
            [[const[i1[0]][i1[1]][i2[0]][i2[1]] for i2 in ids] for i1 in ids]
        )

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
                if self._elastic_constants[i - 1][j - 1] > 1e-10:
                    str1 = "{:.6f}".format(self._elastic_constants[i - 1][j - 1])
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
