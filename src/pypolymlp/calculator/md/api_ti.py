"""API Class for performing thermodynamic integration."""

# import os
from typing import Literal

import numpy as np

from pypolymlp.calculator.md.api_md import PolymlpMD

# from pypolymlp.calculator.md.ase_md import IntegratorASE
from pypolymlp.calculator.md.md_utils import (
    calc_integral,
    calculate_fc2_free_energy,
    find_reference,
    get_p_roots,
    save_thermodynamic_integration_yaml,
)

# import yaml
# from ase.calculators.calculator import Calculator


# from pypolymlp.calculator.properties import Properties

# from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
# from pypolymlp.calculator.utils.io_utils import print_pot
# from pypolymlp.core.data_format import PolymlpStructure
# from pypolymlp.core.interface_vasp import Poscar
# from pypolymlp.core.params import PolymlpParams
# from pypolymlp.core.units import Avogadro, Kb
# from pypolymlp.utils.structure_utils import supercell


class PolymlpTI:
    """API Class for performing thermodynamic integration."""

    def __init__(self, md: PolymlpMD, verbose: bool = False):
        """Init method."""
        self._md = md
        self._verbose = verbose
        self._check_md_instance()

        # self._fc2file = None
        self._log_ti = None
        self._free_energy = None
        self._free_energy_order1 = None
        self._delta_heat_capacity = None

        self._total_free_energy = None
        self._total_free_energy_order1 = None

        self._ref_free_energy = None

        if self._verbose:
            np.set_printoptions(legacy="1.21")

    def _check_md_instance(self):
        """Check required properties in PolymlpMD instance."""
        if self._md.calculator is None:
            raise RuntimeError("Set calculator.")
        if self._md.unitcell is None:
            raise RuntimeError("Set structure.")
        if not self._md.use_reference:
            raise RuntimeError("Set calculator with reference state.")
        return self

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
        heat_capacity: bool
            Calculate heat capacity.
        """
        alphas, weights = get_p_roots(n=n_alphas, a=0.0, b=max_alpha)
        log_ti = []
        for alpha in alphas:
            self.alpha = alpha
            self.run_md_nvt(
                thermostat=thermostat,
                temperature=temperature,
                time_step=time_step,
                ttime=ttime,
                friction=friction,
                n_eq=n_eq,
                n_steps=n_steps,
                interval_log=None,
                logfile=None,
            )
            log_append = self._get_log(alpha)
            log_ti.append(log_append)

        self._log_ti = log_ti = np.array(log_ti)
        de = log_ti[:, 1]
        self._free_energy = calc_integral(weights, de, a=0.0, b=max_alpha)

        self._set_reference_free_energy()
        self._total_free_energy += self._free_energy
        self._total_free_energy_order1 += self._free_energy

        self.alpha = 0.0
        self.run_md_nvt(
            thermostat=thermostat,
            temperature=temperature,
            time_step=time_step,
            ttime=ttime,
            friction=friction,
            n_eq=n_eq,
            n_steps=n_steps * 3,
            interval_log=None,
            logfile=None,
        )
        log_prepend = self._get_log(self.alpha)

        self.alpha = max_alpha
        self.run_md_nvt(
            thermostat=thermostat,
            temperature=temperature,
            time_step=time_step,
            ttime=ttime,
            friction=friction,
            n_eq=n_eq,
            n_steps=n_steps * 3,
            interval_log=None,
            logfile=None,
        )
        log_append = self._get_log(self.alpha)
        self._log_ti = np.vstack([log_prepend, self._log_ti, log_append])

        if self._verbose:
            print("-------------------------------------------", flush=True)
            print("Results (TI):", flush=True)
            np.set_printoptions(suppress=True)
            print("  free_energy:", self._free_energy, flush=True)
            print("  energies (E - E_ref):", flush=True)
            print(log_ti[:, [0, 1]])
            print("-------------------------------------------", flush=True)

        return self

    def _set_reference_free_energy(self):
        """Set reference free energy."""
        if self._total_free_energy is None:
            self._total_free_energy = self._integrator.static_energy
            if self._fc2file is not None:
                self._ref_free_energy = calculate_fc2_free_energy(
                    self._unitcell,
                    self._supercell_matrix,
                    self._fc2file,
                    self._integrator._temperature,
                )
                self._total_free_energy += self._ref_free_energy
                self._total_free_energy_order1 = self._total_free_energy
            else:
                raise RuntimeError("Reference free energy not given.")
        return self

    def _get_log(self, alpha: float):
        """Set log array."""
        log_alpha = [
            alpha,
            self.average_delta_energy_10,
            self.average_energy,
            self.average_total_energy,
            self.average_displacement,
            self.average_delta_energy_1a,
            self.free_energy_perturb,
        ]
        return np.array(log_alpha)

    def save_thermodynamic_integration_yaml(self, filename: str = "polymlp_ti.yaml"):
        """Save results of thermodynamic integration."""
        if not self._use_reference:
            raise RuntimeError("Reference state not found in Calculator.")

        reference = {
            "unitcell": self._unitcell,
            "supercell_matrix": self._supercell_matrix,
            "polymlp": self._pot,
            "fc2_file": self._fc2file,
            "free_energy": self._ref_free_energy,
        }
        save_thermodynamic_integration_yaml(
            self._integrator,
            self._total_free_energy,
            self._free_energy,
            self._log_ti,
            reference,
            delta_heat_capacity=self._delta_heat_capacity,
            filename=filename,
        )
        return self

    def find_reference(self, path_fc2: str, target_temperature: float):
        """Find reference FC2 automatically."""
        return find_reference(path_fc2, target_temperature)


#    @property
#    def delta_energies_10(self):
#        """Return potential energies from reference state."""
#        return self._integrator.delta_energies_10
#
#    @property
#    def delta_energies_1a(self):
#        """Return potential energies from alpha state."""
#        return self._integrator.delta_energies_1a
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
#
#    @property
#    def free_energy_perturb(self):
#        """Return delta free energy from FE perturbation.
#
#        Return delta F = - (1/beta) * ln [<exp(- beta * (E - E_alpha))>_alpha].
#        """
#        return self._integrator.free_energy_perturb
#
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

#    def run_free_energy_perturbation(
#        self,
#        thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
#        temperature: int = 300,
#        time_step: float = 1.0,
#        ttime: float = 20.0,
#        friction: float = 0.01,
#        n_eq: int = 5000,
#        n_steps: int = 20000,
#    ):
#        """Run free energy perturbation.
#
#        Calculate two perturbed values of free energy using ensemble with alpha.
#        free_energy:
#            delta F = - (1 / beta) * ln [<exp(- beta * (E - E_alpha))>_alpha].
#        free_energy_order1:
#            delta F = <E - E_alpha>_alpha.
#
#        Parameters
#        ----------
#        thermostat: Thermostat.
#        temperature : int
#            Target temperature (K).
#        time_step : float
#            Time step for MD (fs).
#        ttime : float
#            Timescale of the Nose-Hoover thermostat (fs).
#        friction : float
#            Friction coefficient for Langevin thermostat (1/fs).
#        n_eq : int
#            Number of equilibration steps.
#        n_steps : int
#            Number of production steps.
#        heat_capacity: bool
#            Calculate heat capacity.
#        """
#        if self._verbose:
#            print("Run free energy perturbation.", flush=True)
#
#        self.run_md_nvt(
#            thermostat=thermostat,
#            temperature=temperature,
#            time_step=time_step,
#            ttime=ttime,
#            friction=friction,
#            n_eq=n_eq,
#            n_steps=n_steps,
#            interval_log=None,
#            logfile=None,
#        )
#        self._set_reference_free_energy()
#        self._free_energy = self.free_energy_perturb
#        self._free_energy_order1 = self.average_delta_energy_1a
#        self._total_free_energy += self._free_energy
#        self._total_free_energy_order1 += self._free_energy_order1
#
#        if self._verbose:
#            print("-------------------------------------------", flush=True)
#            print("Results (Free energy perturbation):", flush=True)
#            np.set_printoptions(suppress=True)
#            print("  free_energy:       ", self._free_energy, flush=True)
#            print("  free_energy_order1:", self._free_energy_order1, flush=True)
#            print("-------------------------------------------", flush=True)
#
#        return self
#
#
#
