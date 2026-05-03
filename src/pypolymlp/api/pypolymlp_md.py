"""API Class for performing MD simulations."""

from typing import Literal, Optional, Union

import numpy as np
from ase.calculators.calculator import Calculator

from pypolymlp.calculator.md.api_md import PolymlpMD
from pypolymlp.calculator.md.api_ti import PolymlpTI
from pypolymlp.calculator.properties import Properties, initialize_polymlp_calculator
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.params import PolymlpParams


class PypolymlpMD:
    """API Class for performing MD simulations."""

    def __init__(self, verbose: bool = False):
        """Init method."""

        self._md = PolymlpMD(verbose=verbose)
        self._ti = None
        self._verbose = verbose
        self._properties = None

        if self._verbose:
            np.set_printoptions(legacy="1.21")

    def _set_polymlp(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
    ):
        """Set parameters on polynomial MLP.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        properties = initialize_polymlp_calculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
        )
        return properties

    def set_ase_calculator(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
    ):
        """Set ASE calculator with polymlp.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        self._properties = self._set_polymlp(pot, params, coeffs, properties)
        self._md.set_ase_calculator(self._properties)
        return self

    def set_ase_calculator_with_fc2(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        fc2hdf5: str = "fc2.hdf5",
        alpha: float = 0.0,
    ):
        """Set ASE calculator using difference between pypolymlp and fc2.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.
        alpha: Mixing parameter. E = alpha * E_polymlp + (1 - alpha) * E_fc2
        fc2hdf5: HDF5 file for second-order force constants.
        """
        self._properties = self._set_polymlp(pot, params, coeffs, properties)
        self._md.set_ase_calculator_with_fc2(
            properties=self._properties,
            fc2hdf5=fc2hdf5,
            alpha=alpha,
        )
        return self

    def set_ase_calculator_with_reference(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        pot_ref: Union[str, list[str]] = None,
        params_ref: Union[PolymlpParams] = None,
        coeffs_ref: Union[np.ndarray, list[np.ndarray]] = None,
        properties_ref: Optional[Properties] = None,
        alpha: float = 0.0,
    ):
        """Set ASE calculator using difference between two pypolymlps.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.
        pot_ref: polymlp file for reference state.
        params_ref: Parameters for polymlp for reference state.
        coeffs_ref: Polymlp coefficients for reference state.
        properties_ref: Properties object for reference state.
        alpha: Mixing parameter. E = alpha * E_polymlp + (1 - alpha) * E_polymlp_ref
        """
        self._properties = self._set_polymlp(pot, params, coeffs, properties)
        properties_ref = self._set_polymlp(
            pot_ref, params_ref, coeffs_ref, properties_ref
        )
        self._md.set_ase_calculator_with_reference(
            properties=self._properties,
            properties_ref=properties_ref,
            alpha=alpha,
        )
        return self

    def set_ase_calculator_with_general_reference(
        self,
        pot_final: Union[str, list[str]] = None,
        params_final: Union[PolymlpParams] = None,
        coeffs_final: Union[np.ndarray, list[np.ndarray]] = None,
        properties_final: Optional[Properties] = None,
        pot_ref: Union[str, list[str]] = None,
        params_ref: Union[PolymlpParams] = None,
        coeffs_ref: Union[np.ndarray, list[np.ndarray]] = None,
        properties_ref: Optional[Properties] = None,
        fc2hdf5: str = "fc2.hdf5",
        alpha_final: float = 0.0,
        alpha_ref: float = 0.0,
        alpha: float = 0.0,
    ):
        """Set ASE calculator using difference between two pypolymlps.

        Parameters
        ----------
        pot_final: polymlp file for final state.
        params_final: Parameters for polymlp for final state.
        coeffs_final: Polymlp coefficients for final state.
        properties_final: Properties object for final state.

        pot_ref: polymlp file for reference state.
        params_ref: Parameters for polymlp for reference state.
        coeffs_ref: Polymlp coefficients for reference state.
        properties_ref: Properties object for reference state.

        fc2hdf5: FC2 HDF file.
        alpha_ref: Mixing parameter for defining reference state.
            E = alpha * E_polymlp_ref + (1 - alpha_ref) * E_fc2
        alpha_final: Mixing parameter for defining final state.
            E = alpha * E_polymlp_final + (1 - alpha_final) * E_fc2
        alpha: Mixing parameter.
            E = alpha * E_final + (1 - alpha) * E_ref
        """
        self._properties = self._set_polymlp(
            pot_final, params_final, coeffs_final, properties_final
        )
        properties_ref = self._set_polymlp(
            pot_ref, params_ref, coeffs_ref, properties_ref
        )
        self._md.set_ase_calculator_with_general_reference(
            properties_final=self._properties,
            properties_ref=properties_ref,
            fc2hdf5=fc2hdf5,
            alpha_final=alpha_final,
            alpha_ref=alpha_ref,
            alpha=alpha,
        )
        return self

    def load_poscar(self, poscar: str):
        """Parse POSCAR file and supercell matrix."""
        self._md.load_poscar(poscar)
        return self

    def set_supercell(self, size: tuple):
        """Set supercell from unitcell.

        Parameter
        ---------
        size: Supercell size with three elements.
              Diagonal elements of supercell matrix.
        """
        self._md.set_supercell(size)
        return self

    def run_md_nvt(
        self,
        thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
        temperature: int = 300,
        time_step: float = 1.0,
        friction: float = 0.01,
        ttime: float = 20.0,
        n_eq: int = 5000,
        n_steps: int = 20000,
        interval_save_forces: Optional[int] = None,
        interval_save_trajectory: Optional[int] = None,
        interval_log: int = 1,
        logfile: str = "log.dat",
        initialize: bool = True,
    ):
        """Run NVT-MD simulation using Nose-Hoover thermostat.

        Parameters
        ----------
        temperature : int
            Target temperature (K).
        time_step : float
            Time step for MD (fs).
        friction : float
            Friction coefficient (1/fs).
        ttime : float
            Timescale of the Nose-Hoover thermostat (fs).
        n_eq : int
            Number of equilibration steps.
        n_steps : int
            Number of production steps.
        """

        if thermostat == "Nose-Hoover":
            self._md.run_Nose_Hoover_NVT(
                temperature=temperature,
                time_step=time_step,
                ttime=ttime,
                n_eq=n_eq,
                n_steps=n_steps,
                interval_save_forces=interval_save_forces,
                interval_save_trajectory=interval_save_trajectory,
                interval_log=interval_log,
                logfile=logfile,
                initialize=initialize,
            )
        elif thermostat == "Langevin":
            self._md.run_Langevin(
                temperature=temperature,
                time_step=time_step,
                friction=friction,
                n_eq=n_eq,
                n_steps=n_steps,
                interval_save_forces=interval_save_forces,
                interval_save_trajectory=interval_save_trajectory,
                interval_log=interval_log,
                logfile=logfile,
                initialize=initialize,
            )
        return self

    def run_free_energy_perturbation(
        self,
        thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
        temperature: int = 300,
        time_step: float = 1.0,
        ttime: float = 20.0,
        friction: float = 0.01,
        n_eq: int = 5000,
        n_steps: int = 20000,
    ):
        """Run free energy perturbation.

        Calculate two perturbed values of free energy using ensemble with alpha.
        free_energy:
            delta F = - (1 / beta) * ln [<exp(- beta * (E - E_alpha))>_alpha].
        free_energy_order1:
            delta F = <E - E_alpha>_alpha.

        Parameters
        ----------
        thermostat: Thermostat.
        temperature : int
            Target temperature (K).
        time_step : float
            Time step for MD (fs).
        ttime : float
            Timescale of the Nose-Hoover thermostat (fs).
        friction : float
            Friction coefficient for Langevin thermostat (1/fs).
        n_eq : int
            Number of equilibration steps.
        n_steps : int
            Number of production steps.

        Return
        ------
        free_energy: Free energy difference in exact form from state alpha.
        free_energy_order1: First-order free energy difference from state alpha.
        """
        free_energy, free_energy_order1 = self._md.run_free_energy_perturbation(
            thermostat=thermostat,
            temperature=temperature,
            time_step=time_step,
            ttime=ttime,
            friction=friction,
            n_eq=n_eq,
            n_steps=n_steps,
        )
        return free_energy, free_energy_order1

    def save_yaml(self, filename: str = "polymlp_md.yaml"):
        """Save properties to yaml file."""
        self._md.save_yaml(filename=filename)
        return self

    def find_reference(self, path_fc2: str, target_temperature: float):
        """Find reference FC2 automatically."""
        return self._md.find_reference(path_fc2, target_temperature)

    def run_thermodynamic_integration(
        self,
        thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
        n_alphas: int = 15,
        max_alpha: float = 1.0,
        temperature: int = 300,
        time_step: float = 1.0,
        ttime: float = 20.0,
        friction: float = 0.01,
        n_eq: int = 5000,
        n_steps: int = 20000,
    ):
        """Run thermodynamic integration.

        Parameters
        ----------
        thermostat: Thermostat.
        n_alphas: Number of sample points for thermodynamic integration
                  using Gaussian quadrature.
        temperature : int
            Target temperature (K).
        time_step : float
            Time step for MD (fs).
        ttime : float
            Timescale of the Nose-Hoover thermostat (fs).
        friction : float
            Friction coefficient for Langevin thermostat (1/fs).
        n_eq : int
            Number of equilibration steps.
        n_steps : int
            Number of production steps.
        """
        self._ti = PolymlpTI(self._md, verbose=self._verbose)
        self._ti.run_thermodynamic_integration(
            thermostat=thermostat,
            n_alphas=n_alphas,
            max_alpha=max_alpha,
            temperature=temperature,
            time_step=time_step,
            ttime=ttime,
            friction=friction,
            n_eq=n_eq,
            n_steps=n_steps,
        )

        return self

    def save_ti_yaml(self, filename: str = "polymlp_ti.yaml"):
        """Save TI properties to yaml file."""
        self._ti.save_ti_yaml(filename=filename)
        return self

    @property
    def unitcell(self):
        """Return unitcell."""
        return self._md.unitcell

    @unitcell.setter
    def unitcell(self, cell: PolymlpStructure):
        """Set unitcell."""
        self._md.unitcell = cell

    @property
    def supercell(self):
        """Return supercell."""
        return self._md.supercell

    @supercell.setter
    def supercell(self, cell: PolymlpStructure):
        """Set supercell."""
        self._md.supercell = cell

    @property
    def final_structure(self):
        """Return structure at the final step."""
        return self._md.final_structure

    @property
    def calculator(self):
        """Return ASE calculator using polymlp and references."""
        return self._md.calculator

    @calculator.setter
    def calculator(self, calc: Calculator):
        """Set calculator."""
        self._md.calculator = calc

    @property
    def energies(self):
        """Return potential energies."""
        return self._md.energies

    @property
    def forces(self):
        """Return forces."""
        return self._md.forces

    @property
    def trajectory(self):
        """Return trajectory."""
        return self._md.trajectory

    @property
    def average_energy(self):
        """Return avarage energy."""
        return self._md.average_energy

    @property
    def average_total_energy(self):
        """Return avarage total energy."""
        return self._md.average_total_energy

    @property
    def heat_capacity(self):
        """Return heat capacity."""
        return self._md.heat_capacity

    @property
    def average_displacement(self):
        """Return avarage energy."""
        return self._md.average_displacement
