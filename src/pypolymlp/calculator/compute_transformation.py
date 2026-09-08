"""Class for calculating transformation path."""

import numpy as np

# from pypolymlp.calculator.opt_geometry import GeometryOptimization
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
        self._supercell = None
        self._sd_cell = None
        self._sd_pos = None

    def set_supercell(self, supercell_matrix: np.ndarray):
        """Set supercell."""
        self._supercell = get_supercell(self._base_structure, supercell_matrix)
        return self._supercell
