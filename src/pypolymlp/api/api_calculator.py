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
            verbose=self._verbose,
        )
        return self._prop

    def set_sscha_calculator(
        self,
        unitcell: PolymlpStructure,
        supercell_matrix: np.ndarray,
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
    ):
        """Set PropertiesSSCHA instance."""
        from pypolymlp.calculator.sscha.api_properties import PropertiesSSCHA
        from pypolymlp.calculator.sscha.sscha_params import SSCHAParams

        if self._prop_static is None:
            raise RuntimeError("Static Properties class instance not found.")

        pot = self._prop_static.pot
        sscha_params = SSCHAParams(
            # unitcell=None,
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
        if self._prop is None:
            return None
        return self._prop.pot

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

    def load_poscar(self, poscar: str):
        """Parse POSCAR file."""
        return Poscar(poscar).structure

    def load_poscars(self, poscars: str):
        """Parse POSCAR files."""
        return parse_structures_from_poscars(poscars)
