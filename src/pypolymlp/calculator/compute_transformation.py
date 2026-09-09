"""Class for calculating transformation path."""

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.supercell_utils import get_supercell

# from pypolymlp.utils.vasp_utils import write_poscar_file


class PolymlpTransformation:
    """Class for calculating transformation pathways."""

    def __init__(
        self,
        structure: PolymlpStructure,
        properties: Properties,
        verbose: bool = False,
    ):
        """Init method."""
        self._prop = properties
        self._verbose = verbose

        self._base_structure = structure
        self._supercell = structure
        self._sd_cell = None
        self._sd_pos = None

    def set_supercell(self, supercell_matrix: np.ndarray):
        """Set supercell."""
        if np.array(supercell_matrix).shape != (3, 3):
            raise RuntimeError("Supercell matrix shape is not (3, 3).")

        for i in range(3):
            if np.array(supercell_matrix)[i, i] < 0:
                raise RuntimeError("Found negative diagonal elements of matrix")

        self._supercell = get_supercell(self._base_structure, supercell_matrix)
        return self._supercell

    # def change_structure(self,

    def run_single(self, disp1: float, disp2: float, gtol: float = 1e-4):
        """Run geometry optimization for single displaced structure."""
        self._supercell_disp = self._change_structure(disp1, disp2)
        self._geometry = GeometryOptimization(
            self._supercell_disp,
            self._prop,
            with_sym=False,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
            selective_dynamics_cell=self._sd_cell,
            selective_dynamics_positions=self._sd_pos,
            verbose=False,
        ).run(gtol=gtol)
        return self._geometry
