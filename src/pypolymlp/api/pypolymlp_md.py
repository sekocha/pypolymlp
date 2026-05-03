"""API Class for performing MD simulations."""

from typing import Literal, Optional, Union

import numpy as np
from ase.calculators.calculator import Calculator

from pypolymlp.calculator.md.api_md import PolymlpMD
from pypolymlp.calculator.properties import Properties, initialize_polymlp_calculator
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.params import PolymlpParams


class PypolymlpMD:
    """API Class for performing MD simulations."""

    def __init__(self, verbose: bool = False):
        """Init method."""

        self._md = PolymlpMD(verbose=verbose)
        self._verbose = verbose
        self._properties = None

        self._use_reference = False
        self._fc2file = None

        # self._log_ti = None
        # self._free_energy = None
        # self._free_energy_order1 = None
        # self._delta_heat_capacity = None
        # self._total_free_energy = None
        # self._total_free_energy_order1 = None
        # self._ref_free_energy = None

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
        self._use_reference = True
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
        self._use_reference = True
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
        self._use_reference = True
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

    #     def _set_reference_free_energy(self):
    #         """Set reference free energy."""
    #         if self._total_free_energy is None:
    #             self._total_free_energy = self._integrator.static_energy
    #             if self._fc2file is not None:
    #                 self._ref_free_energy = calculate_fc2_free_energy(
    #                     self._unitcell,
    #                     self._supercell_matrix,
    #                     self._fc2file,
    #                     self._integrator._temperature,
    #                 )
    #                 self._total_free_energy += self._ref_free_energy
    #                 self._total_free_energy_order1 = self._total_free_energy
    #             else:
    #                 raise RuntimeError("Reference free energy not given.")
    #         return self
    #
    #     def run_thermodynamic_integration(
    #         self,
    #         thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
    #         n_alphas: int = 15,
    #         max_alpha: float = 1.0,
    #         temperature: int = 300,
    #         time_step: float = 1.0,
    #         ttime: float = 20.0,
    #         friction: float = 0.01,
    #         n_eq: int = 5000,
    #         n_steps: int = 20000,
    #         heat_capacity: bool = False,
    #     ):
    #         """Run thermodynamic integration.
    #
    #         Parameters
    #         ----------
    #         thermostat: Thermostat.
    #         n_alphas: Number of sample points for thermodynamic integration
    #                   using Gaussian quadrature.
    #         temperature : int
    #             Target temperature (K).
    #         time_step : float
    #             Time step for MD (fs).
    #         ttime : float
    #             Timescale of the Nose-Hoover thermostat (fs).
    #         friction : float
    #             Friction coefficient for Langevin thermostat (1/fs).
    #         n_eq : int
    #             Number of equilibration steps.
    #         n_steps : int
    #             Number of production steps.
    #         heat_capacity: bool
    #             Calculate heat capacity.
    #         """
    #         if not self._use_reference:
    #             raise RuntimeError("Reference state not found in Calculator.")
    #
    #         alphas, weights = get_p_roots(n=n_alphas, a=0.0, b=max_alpha)
    #         log_ti = []
    #         for alpha in alphas:
    #             self.alpha = alpha
    #             self.run_md_nvt(
    #                 thermostat=thermostat,
    #                 temperature=temperature,
    #                 time_step=time_step,
    #                 ttime=ttime,
    #                 friction=friction,
    #                 n_eq=n_eq,
    #                 n_steps=n_steps,
    #                 interval_log=None,
    #                 logfile=None,
    #             )
    #             log_append = self._get_log(alpha)
    #             log_ti.append(log_append)
    #
    #         self._log_ti = log_ti = np.array(log_ti)
    #         de = log_ti[:, 1]
    #         self._free_energy = calc_integral(weights, de, a=0.0, b=max_alpha)
    #
    #         self._set_reference_free_energy()
    #         self._total_free_energy += self._free_energy
    #         self._total_free_energy_order1 += self._free_energy
    #
    #         self.alpha = 0.0
    #         self.run_md_nvt(
    #             thermostat=thermostat,
    #             temperature=temperature,
    #             time_step=time_step,
    #             ttime=ttime,
    #             friction=friction,
    #             n_eq=n_eq,
    #             n_steps=n_steps * 3,
    #             interval_log=None,
    #             logfile=None,
    #         )
    #         log_prepend = self._get_log(self.alpha)
    #
    #         self.alpha = max_alpha
    #         self.run_md_nvt(
    #             thermostat=thermostat,
    #             temperature=temperature,
    #             time_step=time_step,
    #             ttime=ttime,
    #             friction=friction,
    #             n_eq=n_eq,
    #             n_steps=n_steps * 3,
    #             interval_log=None,
    #             logfile=None,
    #         )
    #         log_append = self._get_log(self.alpha)
    #         self._log_ti = np.vstack([log_prepend, self._log_ti, log_append])
    #
    #         if heat_capacity:
    #             if np.isclose(temperature, 0.0):
    #                 self._delta_heat_capacity = 0.0
    #             else:
    #                 self.alpha = max_alpha
    #                 self.run_md_nvt(
    #                     thermostat=thermostat,
    #                     temperature=temperature,
    #                     time_step=time_step,
    #                     ttime=ttime,
    #                     friction=friction,
    #                     n_eq=n_eq,
    #                     n_steps=n_steps * 10,
    #                     interval_log=None,
    #                     logfile=None,
    #                 )
    #                 self._delta_heat_capacity = self.heat_capacity - 1.5 * Kb * Avogadro
    #
    #         if self._verbose:
    #             print("-------------------------------------------", flush=True)
    #             print("Results (TI):", flush=True)
    #             np.set_printoptions(suppress=True)
    #             print("  free_energy:", self._free_energy, flush=True)
    #             print("  energies (E - E_ref):", flush=True)
    #             print(log_ti[:, [0, 1]])
    #             if heat_capacity:
    #                 print("  delta_heat_capacity:", self._delta_heat_capacity, flush=True)
    #             print("-------------------------------------------", flush=True)
    #
    #         return self
    #
    #     def run_free_energy_perturbation(
    #         self,
    #         thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
    #         temperature: int = 300,
    #         time_step: float = 1.0,
    #         ttime: float = 20.0,
    #         friction: float = 0.01,
    #         n_eq: int = 5000,
    #         n_steps: int = 20000,
    #     ):
    #         """Run thermodynamic integration.
    #
    #         Parameters
    #         ----------
    #         thermostat: Thermostat.
    #         temperature : int
    #             Target temperature (K).
    #         time_step : float
    #             Time step for MD (fs).
    #         ttime : float
    #             Timescale of the Nose-Hoover thermostat (fs).
    #         friction : float
    #             Friction coefficient for Langevin thermostat (1/fs).
    #         n_eq : int
    #             Number of equilibration steps.
    #         n_steps : int
    #             Number of production steps.
    #         heat_capacity: bool
    #             Calculate heat capacity.
    #         """
    #         if not self._use_reference:
    #             raise RuntimeError("Reference state not found in Calculator.")
    #
    #         if self._verbose:
    #             print("Run free energy perturbation.", flush=True)
    #
    #         self.run_md_nvt(
    #             thermostat=thermostat,
    #             temperature=temperature,
    #             time_step=time_step,
    #             ttime=ttime,
    #             friction=friction,
    #             n_eq=n_eq,
    #             n_steps=n_steps,
    #             interval_log=None,
    #             logfile=None,
    #         )
    #         self._set_reference_free_energy()
    #         self._free_energy = self.free_energy_perturb
    #         self._free_energy_order1 = self.average_delta_energy_1a
    #         self._total_free_energy += self._free_energy
    #         self._total_free_energy_order1 += self._free_energy_order1
    #
    #         if self._verbose:
    #             print("-------------------------------------------", flush=True)
    #             print("Results (Free energy perturbation):", flush=True)
    #             np.set_printoptions(suppress=True)
    #             print("  free_energy:       ", self._free_energy, flush=True)
    #             print("  free_energy_order1:", self._free_energy_order1, flush=True)
    #             print("-------------------------------------------", flush=True)
    #
    #         return self
    #
    #     def _get_log(self, alpha: float):
    #         """Set log array."""
    #         log_alpha = [
    #             alpha,
    #             self.average_delta_energy_10,
    #             self.average_energy,
    #             self.average_total_energy,
    #             self.average_displacement,
    #             self.average_delta_energy_1a,
    #             self.free_energy_perturb,
    #         ]
    #         return np.array(log_alpha)
    #
    #     def _check_requisites(self):
    #         """Check requisites for MD simulations."""
    #         if self._supercell_ase is None:
    #             raise RuntimeError("Supercell not found.")
    #         if self._calculator is None:
    #             raise RuntimeError("Calculator not found.")
    #
    #     def _write_conditions(self):
    #         """Write conditions as standard output."""
    #         self._integrator.write_conditions()
    #         return self
    #
    #    def save_thermodynamic_integration_yaml(self, filename: str = "polymlp_ti.yaml"):
    #        """Save results of thermodynamic integration."""
    #        if not self._use_reference:
    #            raise RuntimeError("Reference state not found in Calculator.")
    #
    #        reference = {
    #            "unitcell": self._unitcell,
    #            "supercell_matrix": self._supercell_matrix,
    #            "polymlp": self._pot,
    #            "fc2_file": self._fc2file,
    #            "free_energy": self._ref_free_energy,
    #        }
    #        save_thermodynamic_integration_yaml(
    #            self._integrator,
    #            self._total_free_energy,
    #            self._free_energy,
    #            self._log_ti,
    #            reference,
    #            delta_heat_capacity=self._delta_heat_capacity,
    #            filename=filename,
    #        )
    #        return self
    #
    #    def find_reference(self, path_fc2: str, target_temperature: float):
    #        """Find reference FC2 automatically."""
    #        return find_reference(path_fc2, target_temperature)

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


#    @property
#    def delta_energies_10(self):
#        """Return potential energies from reference state."""
#        return self._md.delta_energies_10
#
#    @property
#    def delta_energies_1a(self):
#        """Return potential energies from alpha state."""
#        return self._md.delta_energies_1a
#
#    @property
#    def average_delta_energy_10(self):
#        """Return avarage delta energy.
#
#        Return <E - E_ref>_alpha.
#        """
#        return self._integrator.average_delta_energy_10
#
#    @property
#    def average_delta_energy_1a(self):
#        """Return avarage delta energy from state alpha.
#
#        Return delta F = <E - E_alpha>_alpha.
#        """
#        return self._integrator.average_delta_energy_1a

#    @property
#    def free_energy_perturb(self):
#        """Return delta free energy from FE perturbation.
#
#        Return delta F = - (1/beta) * ln [<exp(- beta * (E - E_alpha))>_alpha].
#        """
#        return self._integrator.free_energy_perturb

#    @property
#    def total_free_energy(self):
#        """Return total free energy (static + reference + TI)."""
#        return self._total_free_energy
#
#    @property
#    def total_free_energy_order1(self):
#        """Return total free energy (static + ref. + TI + 1st-order perturbation)."""
#        return self._total_free_energy_order1
#
#    @property
#    def reference_free_energy(self):
#        """Return reference free energy."""
#        return self._ref_free_energy
#
#    @property
#    def free_energy(self):
#        """Return difference of free energy from reference state."""
#        return self._free_energy
#
#    @property
#    def free_energy_order1(self):
#        """Return 1st order difference of free energy from reference state."""
#        return self._free_energy_order1
#
#    @property
#    def delta_heat_capacity(self):
#        """Return difference of heat capacity from reference state."""
#        return self._delta_heat_capacity


# def run_thermodynamic_integration(
#     pot: str = "polymlp.yaml",
#     pot_ref: Optional[str] = None,
#     poscar: str = "POSCAR",
#     supercell_size: tuple = (1, 1, 1),
#     thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
#     n_alphas: int = 15,
#     max_alpha: float = 1.0,
#     fc2hdf5: str = "fc2.hdf5",
#     temperature: float = 300.0,
#     time_step: float = 1.0,
#     ttime: float = 20.0,
#     friction: float = 0.01,
#     n_eq: int = 2000,
#     n_steps: int = 20000,
#     filename: str = "polymlp_ti.yaml",
#     heat_capacity: bool = False,
#     verbose: bool = False,
# ):
#     """Run thermodynamic integration.
#
#     Parameters
#     ----------
#     pot: polymlp file.
#     pot_ref: polymlp file for intermediate reference state.
#     poscar: Structure in POSCAR format.
#     supercell_size: Diagonal supercell size.
#     thermostat: Thermostat.
#     n_alphas: Number of sample points for thermodynamic integration
#               using Gaussian quadrature.
#     fc2hdf5: HDF5 file for second-order force constants.
#     temperature : int
#         Target temperature (K).
#     time_step : float
#         Time step for MD (fs).
#     ttime : float
#         Timescale of the Nose-Hoover thermostat (fs).
#     friction : float
#         Friction coefficient for Langevin thermostat (1/fs).
#     n_eq : int
#         Number of equilibration steps.
#     n_steps : int
#         Number of production steps.
#     """
#     pot1 = pot if pot_ref is None else pot_ref
#
#     md = PypolymlpMD(verbose=verbose)
#     md.load_poscar(poscar)
#     md.set_supercell(supercell_size)
#     md.set_ase_calculator_with_fc2(pot=pot1, fc2hdf5=fc2hdf5, alpha=0.0)
#     md.run_thermodynamic_integration(
#         thermostat=thermostat,
#         n_alphas=n_alphas,
#         max_alpha=max_alpha,
#         temperature=temperature,
#         time_step=time_step,
#         ttime=ttime,
#         friction=friction,
#         n_eq=n_eq,
#         n_steps=n_steps,
#         heat_capacity=heat_capacity,
#     )
#     md.save_thermodynamic_integration_yaml(filename=filename)
#
#     if pot_ref is not None:
#         # Path: pot_ref (max_alpha) -> pot (max_alpha) -> pot (1.0)
#         if verbose:
#             print("Path: pot_ref (max_alpha) -- pot_final (max_alpha)", flush=True)
#         free_energy = md.free_energy
#         free_energy1 = md.free_energy
#
#         fep, fep1 = 0.0, 0.0
#         md.set_ase_calculator_with_general_reference(
#             pot_final=pot,
#             pot_ref=pot_ref,
#             fc2hdf5=fc2hdf5,
#             alpha_final=max_alpha,
#             alpha_ref=max_alpha,
#             alpha=0.0,
#         )
#         md.run_free_energy_perturbation(
#             thermostat=thermostat,
#             temperature=temperature,
#             time_step=time_step,
#             ttime=ttime,
#             friction=friction,
#             n_eq=n_eq,
#             n_steps=n_steps,
#         )
#         fep += md.free_energy
#         fep1 += md.free_energy_order1
#
#         if verbose:
#             print("Path: pot_final (max_alpha) -- pot_final (1.0)", flush=True)
#         md.set_ase_calculator_with_fc2(pot=pot, fc2hdf5=fc2hdf5, alpha=max_alpha)
#         md.run_free_energy_perturbation(
#             thermostat=thermostat,
#             temperature=temperature,
#             time_step=time_step,
#             ttime=ttime,
#             friction=friction,
#             n_eq=n_eq,
#             n_steps=n_steps,
#         )
#         fep += md.free_energy
#         fep1 += md.free_energy_order1
#
#         free_energy += fep
#         free_energy1 += fep1
#
#         if verbose:
#             print("FEP delta free energy:              ", fep, flush=True)
#             print("FEP delta free energy (first order):", fep1, flush=True)
#
#         with open(filename, "a") as f:
#             print(file=f)
#             print("free_energy_perturbation_between_polymlps:", file=f)
#
#             print_pot(pot_ref, tag="polymlp_reference", indent=2, file=f)
#             print_pot(pot, tag="polymlp", indent=2, file=f)
#             print("  alpha:              ", 1.0, file=f)
#             print("  free_energy_perturb:", fep, file=f)
#             print("  free_energy:        ", free_energy, file=f)
#             print("  total_free_energy:  ", md.total_free_energy, file=f)
#             print("  first_order:", file=f)
#             print("    free_energy_perturb:", fep1, file=f)
#             print("    free_energy:        ", free_energy1, file=f)
#             print("    total_free_energy:  ", md.total_free_energy_order1, file=f)
#
#     return md
