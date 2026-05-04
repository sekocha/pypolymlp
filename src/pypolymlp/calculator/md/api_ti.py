"""API Class for performing thermodynamic integration."""

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pypolymlp.calculator.md.api_md import PolymlpMD
from pypolymlp.calculator.md.md_utils import calc_integral, get_p_roots
from pypolymlp.calculator.utils.io_utils import print_pot
from pypolymlp.utils.yaml_utils import print_array2d, save_cell


@dataclass
class PropertiesTI:
    """Dataclass of properties for thermodynamic integration step.

    Parameters
    ----------
    average_energy_from_alpha0: <E - E_ref0>_alpha
    free_energy_perturb: - (1/beta) * ln [<exp(- beta * (E - E_alpha))>_alpha]
    free_energy_perturb_order1: <E - E_alpha>_alpha
    """

    alpha: float
    average_energy: float
    average_total_energy: float
    average_displacement: float
    average_energy_from_alpha0: float
    free_energy_perturb: float
    free_energy_perturb_order1: float


class PolymlpTI:
    """API Class for performing thermodynamic integration."""

    def __init__(self, md: PolymlpMD, verbose: bool = False):
        """Init method."""
        self._md = md
        self._verbose = verbose
        self._check_md_instance()

        self._log_ti = None
        self._free_energy = None
        self._energy = None
        self._entropy = None

        self._free_energy_perturb = None
        self._free_energy_perturb_order1 = None

        self._temperature = None

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
        """

        def _run(alpha, const_steps: int = 1):
            self.alpha = alpha
            self._md.run_md_nvt(
                thermostat=thermostat,
                temperature=temperature,
                time_step=time_step,
                ttime=ttime,
                friction=friction,
                n_eq=n_eq,
                n_steps=n_steps * const_steps,
                interval_log=None,
                logfile=None,
            )
            return self._get_log()

        self._temperature = temperature
        alphas, weights = get_p_roots(n=n_alphas, a=0.0, b=max_alpha)
        self._log_ti = [_run(alpha) for alpha in alphas]

        de = np.array([log.average_energy_from_alpha0 for log in self._log_ti])
        self._free_energy = calc_integral(weights, de, a=0.0, b=max_alpha)

        log_prepend = _run(0.0, const_steps=3)
        log_append = _run(max_alpha, const_steps=3)
        self._log_ti = [log_prepend, *self._log_ti, log_append]
        self._energy, self._entropy = self._calc_energy_entropy()

        self._free_energy_perturb = log_append.free_energy_perturb
        self._free_energy_perturb_order1 = log_append.free_energy_perturb_order1

        if self._verbose:
            self._print_log_ti()
        return self

    def _print_log_ti(self):
        """Print log from integration."""
        np.set_printoptions(suppress=True)
        print("-----------------------------------------------------------", flush=True)
        print("Results (Thermodynamic integration):", flush=True)
        print("  unit_energy:  eV/supercell", flush=True)
        print("  unit_entropy: eV/K/supercell", flush=True)
        print("  free_energy:", self._free_energy, flush=True)
        print("  energy:     ", self._energy, flush=True)
        print("  entropy:    ", self._entropy, flush=True)
        print("  de/dalpha <E_max_alpha - E_alpha=0>_alpha:", flush=True)
        for log in self._log_ti:
            alpha = np.round(log.alpha, 7)
            e_alpha0 = np.round(log.average_energy_from_alpha0, 7)
            print("   -", f"{alpha:12}", f"{e_alpha0:12}", flush=True)

        print("-----------------------------------------------------------", flush=True)
        print("Results (Free energy perturbation):", flush=True)
        print("  free_energy (max_alpha -- 1):", self._free_energy_perturb, flush=True)
        val = self._free_energy_perturb_order1
        print("  free_energy (max_alpha -- 1, 1st order):", val, flush=True)
        print("-----------------------------------------------------------", flush=True)
        return self

    def _get_log(self):
        """Set log array."""
        log = PropertiesTI(
            alpha=self.alpha,
            average_energy=self._md.average_energy,
            average_total_energy=self._md.average_total_energy,
            average_displacement=self._md.average_displacement,
            free_energy_perturb=self._md.free_energy_perturb,
            free_energy_perturb_order1=self._md.average_delta_energy_1a,
            average_energy_from_alpha0=self._md.average_delta_energy_10,
        )
        return log

    def _calc_energy_entropy(self):
        """Calculate energy and entropy."""
        if np.isclose(self._temperature, 0.0):
            return 0.0, 0.0

        e_final = self._log_ti[-1].average_energy
        e_ref = self._log_ti[0].average_energy
        energy = e_final - e_ref
        entropy = (energy - self._free_energy) / self._temperature
        return energy, entropy

    @property
    def alpha(self):
        """Return mixing parameter for two states."""
        return self._md.alpha

    @alpha.setter
    def alpha(self, val_alpha: float):
        """Setter of alpha."""
        self._md.alpha = val_alpha

    @property
    def free_energy(self):
        """Return free energy in eV/supercell."""
        return self._free_energy

    @property
    def energy(self):
        """Return energy in eV/supercell."""
        return self._energy

    @property
    def entropy(self):
        """Return entropy in eV/K/supercell."""
        return self._entropy

    @property
    def logs(self):
        """Return properties from each alpha."""
        return self._log_ti

    def save_ti_yaml(self, filename: str = "polymlp_ti.yaml"):
        """Save results of thermodynamic integration."""
        integrator = self._md.integrator
        calculator = self._md.calculator
        np.set_printoptions(legacy="1.21")
        with open(filename, "w") as f:
            print("system:", integrator._atoms.symbols, file=f)
            print(file=f)

            save_cell(self._md.unitcell, tag="unitcell", file=f)
            if self._md.supercell_matrix is not None:
                print_array2d(self._md.supercell_matrix, "supercell_matrix", f)
                print(file=f)

            print("units:", file=f)
            print("  volume:       angstrom3/supercell", file=f)
            print("  temperature:  K", file=f)
            print("  time_step:    fs", file=f)
            print("  energy:       eV/supercell", file=f)
            print("  entropy:      eV/K/supercell", file=f)
            print(file=f)

            print("conditions:", file=f)
            print("  n_atom:     ", len(integrator._atoms.numbers), file=f)
            print("  volume:     ", integrator._atoms.get_volume(), file=f)
            print("  temperature:", integrator._temperature, file=f)
            print("  time_step:  ", integrator._time_step, file=f)
            print("  n_steps_eq: ", integrator._n_eq, file=f)
            print("  n_steps:    ", integrator._n_steps, file=f)
            print("  references: ", os.path.abspath(self._md.fc2file), file=f)

            print_pot(calculator._prop.pot, tag="polymlp", indent=2, file=f)
            print(file=f)

            print("properties:", file=f)
            max_alpha = self._log_ti[-1].alpha
            print("  alpha:           ", max_alpha, file=f)
            print("  free_energy:     ", self._free_energy, file=f)
            print("  entropy:         ", self._entropy, file=f)
            print("  energy:          ", self._energy, file=f)
            print("  static_energy:   ", integrator.static_energy, file=f)
            print(file=f)

            log = self._log_ti[-1]
            print("free_energy_perturbation:", file=f)
            print("  alpha:             ", 1.0, file=f)
            free_energy = self._free_energy + log.free_energy_perturb
            print("  free_energy:       ", free_energy, file=f)
            free_energy = self._free_energy + log.free_energy_perturb_order1
            print("  free_energy_order1:", free_energy, file=f)
            print("  F_perturb:         ", log.free_energy_perturb, file=f)
            print("  F_perturb_order1:  ", log.free_energy_perturb, file=f)
            print(file=f)

            print("sampling_point_properties:", file=f)
            for log in self._log_ti:
                print("- alpha:           ", log.alpha, file=f)
                print("  delta_e:         ", log.average_energy_from_alpha0, file=f)
                print("  energy:          ", log.average_energy, file=f)
                print("  total_energy:    ", log.average_total_energy, file=f)
                print("  F_perturb:       ", log.free_energy_perturb, file=f)
                print("  F_perturb_order1:", log.free_energy_perturb_order1, file=f)
                disp = np.round(log.average_displacement, 5)
                print("  displacement:    ", disp, file=f)
                print(file=f)

        return self


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
