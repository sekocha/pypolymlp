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
        geometry_optimization: bool = True,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure format
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._verbose = verbose
        self._unitcell = unitcell
        fposcar = open(unitcell_poscar)
        self.st_pmg = pmg.core.Structure.from_str(fposcar.read(), fmt="POSCAR")
        fposcar.close()

        self._compute_initial_properties()

    def _compute_initial_properties(self):
        """stress: xx, yy, zz, xy, yz, zx
        --> p0: xx(1), yy(2), zz(3), yz(4), zx(5), xy(6)
        """
        _, _, stress = self.prop.eval(self._unitcell)
        self.eq_stress = -np.array(
            [
                [stress[0], stress[3], stress[5]],
                [stress[3], stress[1], stress[4]],
                [stress[5], stress[4], stress[2]],
            ]
        )

    def run(self):
        """Run elastic constant calculation."""
        deform = DeformedStructureSet(self.st_pmg)
        strains = [d.green_lagrange_strain for d in deform.deformations]

        structure_deform = []
        for i in range(len(deform)):
            structure = copy.deepcopy(self._unitcell)
            lattice = np.array(deform[i].as_dict()["lattice"]["matrix"])
            structure.axis = lattice.T
            structure_deform.append(structure)

        _, _, stresses = self.prop.eval_multiple(structure_deform)
        stresses = convert_stresses_in_gpa(stresses, structure_deform)
        stresses = -np.array(
            [
                [[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]]
                for s in stresses
            ]
        )

        const = diff_fit(strains, stresses, eq_stress=self.eq_stress)[0]

        ids = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        self._elastic_constants = np.array(
            [[const[i1[0]][i1[1]][i2[0]][i2[1]] for i2 in ids] for i1 in ids]
        )

        return self

    def write_elastic_constants(self, filename: str = "polymlp_elastic.yaml"):
        """Save elastic constants."""
        f = open(filename, "w")

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
            if self._elastic_constants[i - 1][j - 1] > 1e-10:
                print(
                    "  c_" + str(i) + str(j) + ":",
                    "{:.6f}".format(self._elastic_constants[i - 1][j - 1]),
                    file=f,
                )
            else:
                print("  c_" + str(i) + str(j) + ": 0", file=f)
        f.close()

    @property
    def elastic_constants(self):
        return self._elastic_constants
