"""API class for SSCHA calculations."""

import os
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.run_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_utils import Restart, SSCHAParameters
from pypolymlp.core.data_format import PolymlpParams
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.phonopy_utils import get_nac_params


class PolymlpSSCHAAPI:
    """API class for SSCHA calculations."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._unitcell = None
        self._supercell_matrix = None
        self._pot = None
        self._prop = None
        self._nac_params = None

        self._sscha_params = None
        self._sscha = None

    def load_poscar(self, poscar: str, supercell_matrix: np.ndarray):
        """Parse POSCAR file and supercell matrix."""
        self._unitcell = Poscar(poscar).structure
        self._supercell_matrix = supercell_matrix
        return self

    def set_polymlp(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
    ):
        """Set MLP.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        if pot is None and params is None and properties is None:
            raise RuntimeError("polymlp not given.")

        self._pot = pot
        if properties is None:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)
        else:
            self._prop = properties
        return self

    def restart(self, yaml: str = "sscha_results.yaml", parse_mlp: bool = True):
        """Parse sscha_results.yaml file."""
        res = Restart(yaml)
        self._unitcell = res.unitcell
        self._supercell_matrix = res.supercell_matrix

        self._pot = res.polymlp
        if parse_mlp and os.path.exists(res.polymlp):
            self._prop = Properties(pot=res.polymlp)
        return self

    def set_nac_params(self, born_vasprun: str):
        """
        born_vasprun: vasprun.xml file for parsing Born effective charges.
        """
        self._nac_params = get_nac_params(
            vasprun=born_vasprun,
            supercell_matrix=self._supercell_matrix,
        )
        return self

    def run(
        self,
        temp: Optional[float] = None,
        temp_min: float = 0,
        temp_max: float = 2000,
        temp_step: float = 50,
        ascending_temp: bool = False,
        n_samples_init: Optional[int] = None,
        n_samples_final: Optional[int] = None,
        tol: float = 0.01,
        max_iter: int = 30,
        mixing: float = 0.5,
        mesh: tuple = (10, 10, 10),
        init_fc_algorithm: Literal["harmonic", "const", "random", "file"] = "harmonic",
        init_fc_file: Optional[str] = None,
    ):
        """Run SSCHA iterations.

        Parameters
        ----------
        Copy from SSCHAParameters.
        """
        if self._prop is None:
            raise RuntimeError("Set polymlp.")
        if self._unitcell is None:
            raise RuntimeError("Set structure.")

        self._sscha_params = SSCHAParameters(
            unitcell=self._unitcell,
            supercell_matrix=self._supercell_matrix,
            pot=self._pot,
            temp=temp,
            temp_min=temp_min,
            temp_max=temp_max,
            temp_step=temp_step,
            ascending_temp=ascending_temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=tol,
            max_iter=max_iter,
            mixing=mixing,
            mesh=mesh,
            init_fc_algorithm=init_fc_algorithm,
            init_fc_file=init_fc_file,
            nac_params=self._nac_params,
        )

        if self._verbose:
            self._sscha_params.print_params()
            self._sscha_params.print_unitcell()

        run_sscha(
            self._sscha_params,
            properties=self._prop,
            verbose=self._verbose,
        )
        return self
