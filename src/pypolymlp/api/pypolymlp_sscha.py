"""API class for SSCHA calculations."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties, initialize_polymlp_calculator
from pypolymlp.calculator.sscha.api_properties import PropertiesSSCHA
from pypolymlp.calculator.sscha.api_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_data import SSCHAData
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.params import PolymlpParams
from pypolymlp.utils.phonopy_utils import get_nac_params


class PypolymlpSSCHA:
    """API class for SSCHA calculations."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._unitcell = None
        self._supercell_matrix = None
        self._pot = None
        self._prop = None
        self._nac_params = None
        self._fc2 = None

        self._sscha_params = None
        self._sscha = None

        if self._verbose:
            np.set_printoptions(legacy="1.21")

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
        self._pot = pot
        self._prop = initialize_polymlp_calculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
        )
        return self

    def load_restart(
        self,
        yaml: str = "sscha_results.yaml",
        parse_fc2: bool = True,
        parse_mlp: bool = True,
        pot: Optional[Union[str, list, tuple, np.ndarray]] = None,
    ):
        """Parse sscha_results.yaml file.

        If parse_fc2 = True, fc2.hdf5 in the same directory
        as yaml file will be loaded.
        """
        if parse_fc2:
            fc2hdf5 = "/".join(yaml.split("/")[:-1]) + "/fc2.hdf5"
        else:
            fc2hdf5 = None
        res = Restart(yaml, fc2hdf5=fc2hdf5)
        self._unitcell = res.unitcell
        self._supercell_matrix = res.supercell_matrix
        if parse_mlp:
            self._pot = res.polymlp if pot is None else pot
            self._prop = Properties(pot=self._pot)
        self._fc2 = res.force_constants
        return self

    def set_nac_params(self, born_vasprun: str):
        """Set parameters for non-analytic corrections.

        Parameters
        ----------
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
        n_temp: Optional[int] = None,
        ascending_temp: bool = False,
        n_samples_init: Optional[int] = None,
        n_samples_final: Optional[int] = None,
        tol: float = 0.005,
        max_iter: int = 50,
        mixing: float = 0.5,
        mesh: tuple = (10, 10, 10),
        init_fc_algorithm: Literal["harmonic", "const", "random", "file"] = "harmonic",
        init_fc_file: Optional[str] = None,
        precondition: bool = True,
        cutoff_radius: Optional[float] = None,
        use_temporal_cutoff: bool = False,
        path: str = "./sscha",
        write_pdos: bool = False,
        use_mkl: bool = True,
    ):
        """Run SSCHA iterations.

        Parameters
        ----------
        temp: Single simulation temperature.
        temp_min: Minimum temperature.
        temp_max: Maximum temperature.
        temp_step: Temperature interval.
        n_temp: Number of temperatures.
                This option is active if n_temp is not None.
                Temperatures are given using Chebyshev nodes.
        ascending_temp: Set simulation temperatures in ascending order.
        n_samples_init: Number of samples in first loop of SSCHA iterations.
                        If None, the number of samples is automatically determined.
        n_samples_final: Number of samples in second loop of SSCHA iterations.
                        If None, the number of samples is automatically determined.
        tol: Convergence tolerance for FCs.
        max_iter: Maximum number of iterations.
        mixing: Mixing parameter.
                FCs are updated by FC2 = FC2(new) * mixing + FC2(old) * (1-mixing).
        mesh: q-point mesh for computing harmonic properties using effective FC2.
        init_fc_algorithm: Algorithm for generating initial FCs.
        init_fc_file: If algorithm = "file", coefficients are read from init_fc_file.
        cutoff_radius: Cutoff radius used for estimating FC2.
        """
        if self._prop is None:
            raise RuntimeError("Set polymlp.")
        if self._unitcell is None:
            raise RuntimeError("Set structure.")
        if self._supercell_matrix is None:
            raise RuntimeError("Set supercell matrix.")

        self._sscha_params = SSCHAParams(
            unitcell=self._unitcell,
            supercell_matrix=self._supercell_matrix,
            pot=self._pot,
            temp=temp,
            temp_min=temp_min,
            temp_max=temp_max,
            temp_step=temp_step,
            n_temp=n_temp,
            ascending_temp=ascending_temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=tol,
            max_iter=max_iter,
            mixing=mixing,
            mesh=mesh,
            init_fc_algorithm=init_fc_algorithm,
            init_fc_file=init_fc_file,
            fc2=self._fc2,
            nac_params=self._nac_params,
            cutoff_radius=cutoff_radius,
            use_mkl=use_mkl,
        )
        if self._verbose:
            self._sscha_params.print_params()
            self._sscha_params.print_unitcell()

        self._sscha = run_sscha(
            self._sscha_params,
            self._prop,
            precondition=precondition,
            use_temporal_cutoff=use_temporal_cutoff,
            path=path,
            write_pdos=write_pdos,
            verbose=self._verbose,
        )
        self._fc2 = self._sscha.force_constants
        return self

    def run_geometry_optimization(
        self,
        temp: float = 1000,
        n_samples_init: Optional[int] = None,
        n_samples_final: Optional[int] = None,
        tol: float = 0.005,
        max_iter: int = 50,
        mixing: float = 0.5,
        mesh: tuple = (10, 10, 10),
        init_fc_algorithm: Literal["harmonic", "const", "random", "file"] = "harmonic",
        init_fc_file: Optional[str] = None,
        cutoff_radius: Optional[float] = None,
        use_mkl: bool = True,
        with_sym: bool = True,
        relax_cell: bool = False,
        relax_volume: bool = False,
        relax_positions: bool = True,
        method: Literal["BFGS", "CG", "L-BFGS-B", "SLSQP"] = "BFGS",
        gtol: float = 2e-2,
        maxiter: int = 1000,
        c1: float = 1e-3,
        c2: float = 0.5,
        pressure: float = 0.0,
    ):
        """Run geometry optimization using SSCHA.

        Parameters
        ----------
        temp: Single simulation temperature.
        n_samples_init: Number of samples in first loop of SSCHA iterations.
                        If None, the number of samples is automatically determined.
        n_samples_final: Number of samples in second loop of SSCHA iterations.
                        If None, the number of samples is automatically determined.
        tol: Convergence tolerance for FCs.
        max_iter: Maximum number of iterations.
        mixing: Mixing parameter.
                FCs are updated by FC2 = FC2(new) * mixing + FC2(old) * (1-mixing).
        mesh: q-point mesh for computing harmonic properties using effective FC2.
        init_fc_algorithm: Algorithm for generating initial FCs.
        init_fc_file: If algorithm = "file", coefficients are read from init_fc_file.
        cutoff_radius: Cutoff radius used for estimating FC2.

        (For geometry optimization)
        with_sym: Consider symmetry.
        relax_cell: Relax cell.
        relax_volume: Relax volume.
        relax_positions: Relax atomic positions.
        method: Optimization method, CG, BFGS, L-BFGS-B, or SLSQP.
                If relax_volume = False, SLSQP is automatically used.
        gtol: Tolerance for gradients.
        maxiter: Maximum iteration in scipy optimization.
        c1: c1 parameter in scipy optimization.
        c2: c2 parameter in scipy optimization.
        pressure: Pressure in GPa.
        """
        if self._prop is None:
            raise RuntimeError("Set polymlp.")
        if self._unitcell is None:
            raise RuntimeError("Set structure.")
        if self._supercell_matrix is None:
            raise RuntimeError("Set supercell matrix.")

        self._sscha_params = SSCHAParams(
            unitcell=self._unitcell,
            supercell_matrix=self._supercell_matrix,
            pot=self._pot,
            temp=temp,
            n_samples_init=n_samples_init,
            n_samples_final=n_samples_final,
            tol=tol,
            max_iter=max_iter,
            mixing=mixing,
            mesh=mesh,
            init_fc_algorithm=init_fc_algorithm,
            init_fc_file=init_fc_file,
            fc2=self._fc2,
            nac_params=self._nac_params,
            cutoff_radius=cutoff_radius,
            use_mkl=use_mkl,
        )
        if self._verbose:
            self._sscha_params.print_params()
            self._sscha_params.print_unitcell()

        prop_sscha = PropertiesSSCHA(
            self._sscha_params,
            self._prop,
            verbose=self._verbose,
        )
        opt = GeometryOptimization(
            self._unitcell,
            prop_sscha,
            relax_cell=relax_cell,
            relax_volume=relax_volume,
            relax_positions=relax_positions,
            with_sym=with_sym,
            pressure=pressure,
            verbose=self._verbose,
        )
        opt.run(method=method, gtol=gtol, maxiter=maxiter, c1=c1, c2=c2)
        opt.write_poscar()
        if self._verbose:
            opt.print_residuals()
            print("Final structure", flush=True)
            opt.print_structure()

        self._fc2 = prop_sscha.force_constants
        return self

    @property
    def unitcell(self):
        """Return unit cell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell: PolymlpStructure):
        """Setter of unit cell."""
        self._unitcell = cell

    @property
    def supercell_matrix(self):
        """Return supercell_matrix."""
        return self._supercell_matrix

    @supercell_matrix.setter
    def supercell_matrix(self, matrix: np.ndarray):
        """Setter of unit cell."""
        self._supercell_matrix = matrix

    @property
    def sscha_params(self) -> SSCHAParams:
        """Return SSCHA parameters."""
        return self._sscha_params

    @property
    def properties(self) -> SSCHAData:
        """Return SSCHA properties at the final temperature."""
        if self._sscha is None:
            return None
        return self._sscha.properties

    @property
    def logs(self) -> list[SSCHAData]:
        """Return logs of SSCHA properties at the final temperature."""
        if self._sscha is None:
            return None
        return self._sscha.logs

    @property
    def force_constants(self) -> np.ndarray:
        """Return FC2 at the final temperature.

        shape=(n_atom, n_atom, 3, 3).
        """
        return self._fc2

    @property
    def sscha_properties(self) -> SSCHAData:
        """Return SSCHA properties at the final temperature.

        Deprecated.
        """
        if self._sscha is None:
            return None
        return self._sscha.properties

    @property
    def sscha_logs(self) -> list[SSCHAData]:
        """Return logs of SSCHA properties at the final temperature.

        Deprecated.
        """
        if self._sscha is None:
            return None
        return self._sscha.logs
