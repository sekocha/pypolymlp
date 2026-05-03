"""API Class for performing MD simulations."""

import os
from typing import Literal, Optional

import numpy as np
import yaml
from ase.calculators.calculator import Calculator

from pypolymlp.calculator.md.ase_md import IntegratorASE
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator
from pypolymlp.calculator.utils.ase_calculator_ref import (
    PolymlpFC2ASECalculator,
    PolymlpGeneralRefASECalculator,
    PolymlpRefASECalculator,
)
from pypolymlp.calculator.utils.ase_utils import (
    ase_atoms_to_structure,
    structure_to_ase_atoms,
)
from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell

# from pypolymlp.core.units import Avogadro, Kb


class PolymlpMD:
    """API Class for performing MD simulations."""

    # TODO: Implement Nose-Hoover-chain thermostat.
    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose

        self._properties = None
        self._calculator = None
        self._integrator = None

        self._unitcell = None
        self._supercell = None
        self._unitcell_ase = None
        self._supercell_ase = None
        self._supercell_matrix = None

        self._use_reference = False
        self._fc2file = None

        if self._verbose:
            np.set_printoptions(legacy="1.21")

    def _set_fc2(self, fc2hdf5: str = "fc2.hdf5"):
        """Set FC2 parameters and check errors in FC2."""
        path = "/".join(os.path.abspath(fc2hdf5).split("/")[:-1])
        sscha_yaml = path + "/sscha_results.yaml"
        if os.path.exists(sscha_yaml):
            yaml_data = yaml.safe_load(open(sscha_yaml))
            if yaml_data["status"]["imaginary"]:
                raise RuntimeError("Given FC2 shows imaginary frequencies.")

        self._fc2file = fc2hdf5
        fc2 = load_fc2_hdf5(fc2hdf5, return_matrix=True)
        return fc2

    def set_ase_calculator(self, properties: Properties):
        """Set ASE calculator with polymlp.

        Parameters
        ----------
        properties: Properties instance.
        """
        self._properties = properties
        self._calculator = PolymlpASECalculator(properties=properties)
        return self._calculator

    def set_ase_calculator_with_fc2(
        self,
        properties: Properties,
        fc2hdf5: str = "fc2.hdf5",
        alpha: float = 0.0,
    ):
        """Set ASE calculator using difference between pypolymlp and fc2.

        Parameters
        ----------
        properties: Properties object.
        alpha: Mixing parameter. E = alpha * E_polymlp + (1 - alpha) * E_fc2
        fc2hdf5: HDF5 file for second-order force constants.
        """
        if self._supercell is None:
            raise RuntimeError("Supercell not found.")

        self._use_reference = True
        self._properties = properties
        fc2 = self._set_fc2(fc2hdf5)
        self._calculator = PolymlpFC2ASECalculator(
            fc2,
            self._supercell,
            properties=properties,
            alpha=alpha,
        )
        return self._calculator

    def set_ase_calculator_with_reference(
        self,
        properties: Properties,
        properties_ref: Properties,
        alpha: float = 0.0,
    ):
        """Set ASE calculator using difference between two pypolymlps.

        Parameters
        ----------
        properties: Properties object.
        properties_ref: Properties object for reference state.
        alpha: Mixing parameter. E = alpha * E_polymlp + (1 - alpha) * E_polymlp_ref
        """
        self._use_reference = True
        self._properties = properties
        self._calculator = PolymlpRefASECalculator(
            properties=properties,
            properties_ref=properties_ref,
            alpha=alpha,
        )
        return self._calculator

    def set_ase_calculator_with_general_reference(
        self,
        properties_final: Optional[Properties] = None,
        properties_ref: Optional[Properties] = None,
        fc2hdf5: str = "fc2.hdf5",
        alpha_final: float = 0.0,
        alpha_ref: float = 0.0,
        alpha: float = 0.0,
    ):
        """Set ASE calculator using difference between two pypolymlps.

        Parameters
        ----------
        properties_final: Properties object for final state.
        properties_ref: Properties object for reference state.

        fc2hdf5: FC2 HDF file.
        alpha_ref: Mixing parameter for defining reference state.
            E = alpha * E_polymlp_ref + (1 - alpha_ref) * E_fc2
        alpha_final: Mixing parameter for defining final state.
            E = alpha * E_polymlp_final + (1 - alpha_final) * E_fc2
        alpha: Mixing parameter.
            E = alpha * E_final + (1 - alpha) * E_ref
        """
        if self._supercell is None:
            raise RuntimeError("Supercell not found.")

        self._use_reference = True
        self._properties = properties_final
        fc2 = self._set_fc2(fc2hdf5)
        self._calculator = PolymlpGeneralRefASECalculator(
            fc2,
            self._supercell,
            properties_final=properties_final,
            properties_ref=properties_ref,
            alpha_final=alpha_final,
            alpha_ref=alpha_ref,
            alpha=alpha,
        )
        return self._calculator

    def load_poscar(self, poscar: str):
        """Parse POSCAR file and supercell matrix."""
        self._unitcell = Poscar(poscar).structure
        self._unitcell_ase = structure_to_ase_atoms(self._unitcell)
        self._supercell = self._unitcell
        self._supercell_ase = self._unitcell_ase
        return self

    def set_supercell(self, size: tuple):
        """Set supercell from unitcell.

        Parameter
        ---------
        size: Supercell size with three elements.
              Diagonal elements of supercell matrix.
        """
        if self._unitcell is None:
            raise RuntimeError("Unitcell not found.")
        if len(size) != 3:
            raise RuntimeError("Supercell size is not equal to 3.")
        self._supercell = supercell(self._unitcell, size)
        self._supercell_ase = structure_to_ase_atoms(self._supercell)
        self._supercell_matrix = np.diag(size)
        return self

    def run_Nose_Hoover_NVT(
        self,
        temperature: int = 300,
        time_step: float = 1.0,
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
        ttime : float
            Timescale of the Nose-Hoover thermostat (fs).
        n_eq : int
            Number of equilibration steps.
        n_steps : int
            Number of production steps.
        """
        self._check_requisites()
        self._integrator = IntegratorASE(
            atoms=self._supercell_ase, calc=self._calculator
        )
        self._integrator.set_integrator_Nose_Hoover_NVT(
            temperature=temperature,
            time_step=time_step,
            ttime=ttime,
            initialize=initialize,
        )
        self._integrator.activate_loggers(
            logfile=logfile,
            interval_log=interval_log,
            interval_save_forces=interval_save_forces,
            interval_save_trajectory=interval_save_trajectory,
        )
        if self._verbose:
            self._write_conditions()
            self._integrator.activate_standard_output(interval=100)

        self._integrator.run(n_eq=n_eq, n_steps=n_steps)
        return self

    def run_Langevin(
        self,
        temperature: int = 300,
        time_step: float = 1.0,
        friction: float = 0.01,
        n_eq: int = 5000,
        n_steps: int = 20000,
        interval_save_forces: Optional[int] = None,
        interval_save_trajectory: Optional[int] = None,
        interval_log: int = 1,
        logfile: str = "log.dat",
        initialize: bool = True,
    ):
        """Run NVT-MD simulation using Langevin dynamics.

        Parameters
        ----------
        temperature : int
            Target temperature (K).
        time_step : float
            Time step for MD (fs).
        friction : float
            Friction coefficient (1/fs).
        n_eq : int
            Number of equilibration steps.
        n_steps : int
            Number of production steps.
        """
        self._check_requisites()
        self._integrator = IntegratorASE(
            atoms=self._supercell_ase, calc=self._calculator
        )
        self._integrator.set_integrator_Langevin(
            temperature=temperature,
            time_step=time_step,
            friction=friction,
            initialize=initialize,
        )
        self._integrator.activate_loggers(
            logfile=logfile,
            interval_log=interval_log,
            interval_save_forces=interval_save_forces,
            interval_save_trajectory=interval_save_trajectory,
        )
        if self._verbose:
            self._write_conditions()
            self._integrator.activate_standard_output(interval=100)

        self._integrator.run(n_eq=n_eq, n_steps=n_steps)
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
        if thermostat == "Nose-Hoover":
            self.run_Nose_Hoover_NVT(
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
            self.run_Langevin(
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
        if not self._use_reference:
            raise RuntimeError("Reference state not defined.")

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
        free_energy = self.free_energy_perturb
        free_energy_order1 = self.average_delta_energy_1a

        if self._verbose:
            print("-------------------------------------------", flush=True)
            print("Free energy perturbation:", flush=True)
            np.set_printoptions(suppress=True)
            print("  free_energy:       ", free_energy, flush=True)
            print("  free_energy_order1:", free_energy_order1, flush=True)
            print("-------------------------------------------", flush=True)

        return (free_energy, free_energy_order1)

    def _check_requisites(self):
        """Check requisites for MD simulations."""
        if self._supercell_ase is None:
            raise RuntimeError("Supercell not found.")
        if self._calculator is None:
            raise RuntimeError("Calculator not found.")

    def _write_conditions(self):
        """Write conditions as standard output."""
        self._integrator.write_conditions()
        return self

    def save_yaml(self, filename: str = "polymlp_md.yaml"):
        """Save properties to yaml file."""
        self._integrator.save_yaml(filename=filename)
        return self

    @property
    def use_reference(self):
        """Return whether reference state is used in calculator."""
        return self._use_reference

    @property
    def unitcell(self):
        """Return unitcell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell: PolymlpStructure):
        """Set unitcell."""
        self._unitcell = cell
        self._unitcell_ase = structure_to_ase_atoms(self._unitcell)

    @property
    def supercell(self):
        """Return supercell."""
        return self._supercell

    @supercell.setter
    def supercell(self, cell: PolymlpStructure):
        """Set supercell."""
        self._supercell = cell
        self._supercell_ase = structure_to_ase_atoms(self._supercell)

    @property
    def final_structure(self):
        """Return structure at the final step."""
        if self._integrator is None:
            return None
        return ase_atoms_to_structure(self._supercell_ase)

    @property
    def calculator(self):
        """Return calculator."""
        return self._calculator

    @calculator.setter
    def calculator(self, calc: Calculator):
        """Set calculator."""
        self._calculator = calc
        if self._integrator is not None:
            self._integrator.calculator = calc

    @property
    def alpha(self):
        """Return mixing parameter for two states."""
        try:
            return self._calculator.alpha
        except:
            return None

    @alpha.setter
    def alpha(self, val_alpha: float):
        if val_alpha < 0.0 or val_alpha > 1.0:
            RuntimeError("alpha must be between 0.0 and 1.0.")
        self._calculator.alpha = val_alpha
        if self._integrator is not None:
            self._integrator.calculator.alpha = val_alpha

    @property
    def energies(self):
        """Return potential energies."""
        if self._integrator is None:
            return None
        return self._integrator.energies

    @property
    def forces(self):
        """Return forces."""
        if self._integrator is None:
            return None
        return self._integrator.forces

    @property
    def trajectory(self):
        """Return trajectory."""
        if self._integrator is None:
            return None
        return [ase_atoms_to_structure(t) for t in self._integrator.trajectory]

    @property
    def average_energy(self):
        """Return avarage energy."""
        if self._integrator is None:
            return None
        return self._integrator.average_energy

    @property
    def average_total_energy(self):
        """Return avarage total energy."""
        if self._integrator is None:
            return None
        return self._integrator.average_total_energy

    @property
    def heat_capacity(self):
        """Return heat capacity."""
        if self._integrator is None:
            return None
        return self._integrator.heat_capacity

    @property
    def average_displacement(self):
        """Return avarage energy."""
        if self._integrator is None:
            return None
        return self._integrator.average_displacement

    @property
    def delta_energies_10(self):
        """Return potential energies from reference state."""
        if self._integrator is None:
            return None
        return self._integrator.delta_energies_10

    @property
    def delta_energies_1a(self):
        """Return potential energies from alpha state."""
        if self._integrator is None:
            return None
        return self._integrator.delta_energies_1a

    @property
    def average_delta_energy_10(self):
        """Return avarage delta energy.

        Return <E - E_ref>_alpha.
        """
        if self._integrator is None:
            return None
        return self._integrator.average_delta_energy_10

    @property
    def average_delta_energy_1a(self):
        """Return avarage delta energy from state alpha.

        Return delta F = <E - E_alpha>_alpha.
        """
        if self._integrator is None:
            return None
        return self._integrator.average_delta_energy_1a

    @property
    def free_energy_perturb(self):
        """Return delta free energy from FE perturbation.

        Return delta F = - (1/beta) * ln [<exp(- beta * (E - E_alpha))>_alpha].
        """
        if self._integrator is None:
            return None
        return self._integrator.free_energy_perturb
