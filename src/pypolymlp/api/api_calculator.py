"""API Class for setting property class instance."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.calculator.properties import Properties, initialize_polymlp_calculator
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar, parse_structures_from_poscars
from pypolymlp.core.params import PolymlpParams


class PypolymlpCalcProperties:
    """API Class for setting property class instance."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        if self._verbose:
            np.set_printoptions(legacy="1.21")

        self._prop = None
        self._prop_static = None
        self._prop_dyn = None

        self._sscha_unitcell = None
        self._sscha_supercell = None
        self._sscha_fc2 = None

    def set_polymlp(
        self,
        pot: Optional[Union[str, list[str]]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        properties: Optional[Properties] = None,
        require_mlp: bool = True,
    ):
        """Set polymlp Properties instance.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        self._prop = self._prop_static = initialize_polymlp_calculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            return_none=not require_mlp,
        )
        return self._prop

    def set_lammps(
        self,
        elements: list,
        pot: str = "polymlp.yaml",
        style: str = "polymlp",
        style_command: str = "pair_style",
        coeff_command: str = "pair_coeff",
        log: bool = False,
        screen: bool = False,
    ):
        """Set PropertiesLammps instance.

        Lammps python module is required.
        """
        from pypolymlp.calculator.utils.lammps.properties_lammps import PropertiesLammps

        self._prop = self._prop_static = PropertiesLammps(
            elements=elements,
            pot=pot,
            style=style,
            style_command=style_command,
            coeff_command=coeff_command,
            log=log,
            verbose=False,
        )
        return self._prop

    def load_sscha_restart(
        self,
        yaml: str = "sscha_results.yaml",
        parse_fc2: bool = True,
        parse_mlp: bool = True,
        pot: Optional[Union[str, list, tuple, np.ndarray]] = None,
    ):
        """Parse sscha_results.yaml file.

        Parameters
        ----------
        yaml: yaml file used for restarting SSCHA.
        parse_fc2: Parse force constants or not.
        parse_mlp: Parse polymlp or not.
        pot: Polymlp file.

        If parse_fc2 = True,
        fc2.hdf5 in the same directory as yaml file will be loaded.
        """
        from pypolymlp.calculator.sscha.api_sscha import load_restart

        (
            self._sscha_unitcell,
            self._sscha_supercell_matrix,
            self._prop_static,
            self._sscha_fc2,
        ) = load_restart(
            yaml=yaml,
            parse_fc2=parse_fc2,
            parse_mlp=parse_mlp,
            pot=pot,
        )
        self._prop = self._prop_static
        return self._prop

    @property
    def sscha_unitcell(self):
        """Unit cell for SSCHA calculation."""
        return self._sscha_unitcell

    @property
    def sscha_supercell_matrix(self):
        """Supercell matrix for SSCHA calculation."""
        return self._sscha_supercell_matrix

    @property
    def sscha_fc2(self):
        """Force constants for SSCHA calculation."""
        return self._sscha_fc2

    def set_sscha_calculator(
        self,
        unitcell: Optional[PolymlpStructure] = None,
        supercell_matrix: Optional[np.ndarray] = None,
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
        fc2: Optional[np.ndarray] = None,
        nac_params: Optional[np.ndarray] = None,
        precondition: bool = True,
        cutoff_radius: Optional[float] = None,
        use_temporal_cutoff: bool = False,
        path: str = "./sscha",
        write_pdos: bool = False,
        use_mkl: bool = True,
        symfc_batch_size: int = 200,
    ):
        """Set PropertiesSSCHA instance.

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
        from pypolymlp.calculator.sscha.api_properties import PropertiesSSCHA
        from pypolymlp.calculator.sscha.sscha_params import SSCHAParams

        if self._prop_static is None:
            raise RuntimeError("Static Properties class instance not found.")

        if unitcell is None:
            unitcell = self._sscha_unitcell

        if supercell_matrix is None:
            supercell_matrix = self._sscha_supercell_matrix

        pot = self._prop_static.pot
        sscha_params = SSCHAParams(
            unitcell=unitcell,
            supercell_matrix=supercell_matrix,
            pot=pot,
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
            fc2=fc2,
            nac_params=nac_params,
            cutoff_radius=cutoff_radius,
            use_mkl=use_mkl,
            symfc_batch_size=symfc_batch_size,
        )
        self._prop = self._prop_dyn = PropertiesSSCHA(
            sscha_params,
            self._prop_static,
            precondition=precondition,
            use_temporal_cutoff=use_temporal_cutoff,
            path=path,
            write_pdos=write_pdos,
            verbose=self._verbose,
        )
        return self._prop

    def eval(self, structure: PolymlpStructure):
        """Evaluate properties for a single structure.

        Returns
        -------
        e: Energy. unit: eV/supercell
        f: Forces. shape=(3, natom), unit: eV/angstrom.
        s: Stress tensors. shape=(6),
            unit: eV/supercell in the order of xx, yy, zz, xy, yz, zx.
        """
        if self._prop is None:
            raise RuntimeError("Properties class instance not found.")
        return self._prop.eval(structure)

    def eval_multiple(self, structures: PolymlpStructure | list[PolymlpStructure]):
        """Evaluate properties for a single structure or multiple structures.

        Returns
        -------
        e: Energy. shape=(n_str,), unit: eV/supercell
        f: Forces. shape=(n_str, 3, natom), unit: eV/angstrom.
        s: Stress tensors. shape=(n_str, 6),
            unit: eV/supercell in the order of xx, yy, zz, xy, yz, zx.
        """
        if self._prop is None:
            raise RuntimeError("Properties class instance not found.")
        return self._prop.eval_multiple(structures)

    def save(self):
        """Save properties.

        Numpy files of polymlp_energies.npy, polymlp_forces.npy,
        and polymlp_stress_tensors.npy are generated.
        They contain the energy values, forces, and stress tensors
        for structures used for the latest run of self.eval.
        """
        if self._prop is not None:
            self._prop.save(verbose=self._verbose)
        return self

    @property
    def calculator(self):
        """Return property class instance."""
        return self._prop

    @property
    def static_calculator(self):
        """Return static property class instance."""
        return self._prop_static

    @property
    def dynamic_calculator(self):
        """Return dynamic property class instance."""
        return self._prop_dyn

    @property
    def elements(self) -> tuple:
        """Return elements."""
        if self._prop is None:
            return None
        return self._prop.elements

    @property
    def pot(self) -> list:
        """Return potential file name."""
        if self._prop_static is None:
            return None
        return self._prop_static.pot

    @property
    def sscha(self):
        """Return SSCHACore instance with results."""
        try:
            return self._prop._sscha
        except:
            return None

    @property
    def force_constants(self) -> np.ndarray:
        """Return force constants."""
        try:
            return self._prop.force_constants
        except:
            return None

    def load_poscar(self, poscar: str):
        """Parse POSCAR file."""
        cell = Poscar(poscar).structure
        self._sscha_unitcell = cell
        return cell

    def load_poscars(self, poscars: str):
        """Parse POSCAR files."""
        return parse_structures_from_poscars(poscars)

    def get_nac_params(self, born_vasprun: str, supercell_matrix: np.ndarray):
        """Return NAC parameters.

        Parameters
        ----------
        born_vasprun: vasprun.xml file for parsing Born effective charges.
        """
        from pypolymlp.utils.phonopy_utils import get_nac_params

        nac_params = get_nac_params(
            vasprun=born_vasprun,
            supercell_matrix=supercell_matrix,
        )
        return nac_params

    @property
    def energies(self) -> np.ndarray:
        """Return energies from the final calculation."""
        try:
            return self._prop.energies
        except:
            return None

    @property
    def forces(self) -> list:
        """Return forces from the final calculation."""
        try:
            return self._prop.forces
        except:
            return None

    @property
    def stresses(self) -> np.ndarray:
        """Return stress tensors from the final calculation."""
        try:
            return self._prop.stresses
        except:
            return None

    @property
    def stresses_gpa(self) -> np.ndarray:
        """Return stress tensors in GPa from the final calculation."""
        try:
            return self._prop.stresses_gpa
        except:
            return None
