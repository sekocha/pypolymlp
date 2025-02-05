"""API Class for performing grid search for finding optimal MLPs."""

from typing import Optional, Union

import numpy as np

# from pypolymlp.core.data_format import PolymlpStructure
# from pypolymlp.mlp_opt.optimal import find_optimal_mlps
# from pypolymlp.utils.count_time import PolymlpCost
# from pypolymlp.utils.dataset.auto_divide import auto_divide
# from pypolymlp.utils.vasp_utils import (
#     load_electronic_properties_from_vasprun,
#     print_poscar,
#     write_poscar_file,
# )
# from pypolymlp.utils.vasprun_compress import convert


class PolymlpGridSearch:
    """API Class for performing grid search for finding optimal MLPs."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
