"""Functions for MD integrators in ASE."""

from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from pypolymlp.core.units import Avogadro, EVtoJ, KbEV


class IntegratorASE:
    """Wrapper of integrators in ASE."""

    def __init__(self, atoms: Atoms, calc: Calculator):
        """Initialize IntegratorASE.

        Parameters
        ----------
        atoms: Initial structure in ASE Atoms.
        calc: ASE calculator used for computing energies and forces.
        """
        self.atoms = atoms
        self.calculator = calc
        self._dyn = None

        self._energies = None
        self._forces = None
        self._trajectory = None

        self._average_energy = None
        self._heat_capacity = None
        self._heat_capacity_eV = None

        if not hasattr(calc, "_use_reference") or not calc._use_reference:
            self._use_reference = False
        else:
            self._use_reference = True
        # Required if reference state is given.
        self._delta_energies = None
        self._displacements = None
        self._average_delta_energy = None
        self._average_displacement = None

        self._temperature = None
        self._time_step = None
        self._n_eq = None
        self._n_steps = None
        self._thermostat = None

    def activate_MDLogger(
        self,
        logfile: str,
        header: bool = True,
        stress: bool = False,
        peratom: bool = True,
        mode: str = "w",
        interval: int = 1,
    ):
        """Attach MDLogger to the current dynamics."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        logger = MDLogger(
            dyn=self._dyn,
            atoms=self._atoms,
            logfile=logfile,
            header=header,
            stress=stress,
            peratom=peratom,
            mode=mode,
        )
        self._dyn.attach(logger, interval=interval)
        return self

    def activate_standard_output(self, interval: int = 100):
        """Attach MDLogger for stdout to the current dynamics."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        logger = MDLogger(
            dyn=self._dyn,
            atoms=self._atoms,
            logfile="-",
            header=True,
            stress=False,
            peratom=True,
        )
        self._dyn.attach(logger, interval=interval)
        return self

    def activate_save_energies(self, interval: int = 1):
        """Save potential energies during MD."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        self._energies = []

        def store_energies_in_memory():
            e = self._atoms.get_potential_energy()
            self._energies.append(e)

        self._dyn.attach(store_energies_in_memory, interval=interval)
        return self

    def activate_save_forces(self, interval: int = 1):
        """Save forces during MD."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        self._forces = []

        def store_forces_in_memory():
            f = self._atoms.get_forces()
            self._forces.append(f)

        self._dyn.attach(store_forces_in_memory, interval=interval)
        return self

    def activate_save_trajectory(self, interval: int = 100):
        """Save trajectory during MD."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        self._trajectory = []

        def store_trajectory_in_memory():
            self._trajectory.append(self._atoms.copy())

        self._dyn.attach(store_trajectory_in_memory, interval=interval)
        return self

    def activate_save_energy_differences(self, interval: int = 1):
        """Save energy difference from reference states."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        if not self._use_reference:
            raise RuntimeError("Reference state not defined in Calculator.")

        self._delta_energies = []

        def store_de_in_memory():
            e = self.calculator.delta_energy
            self._delta_energies.append(e)

        self._dyn.attach(store_de_in_memory, interval=interval)
        return self

    def activate_save_displacement(self, interval: int = 1):
        """Save average displacement."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")

        if not self._use_reference:
            raise RuntimeError("Reference state not defined in Calculator.")

        self._displacements = []

        def store_displacement_in_memory():
            d = self.calculator.average_displacement
            self._displacements.append(d)

        self._dyn.attach(store_displacement_in_memory, interval=interval)
        return self

    def activate_loggers(
        self,
        logfile: str = "log.dat",
        interval_log: int = 1,
        interval_save_forces: Optional[int] = None,
        interval_save_trajectory: Optional[int] = None,
    ):
        self.activate_save_energies(interval=1)
        if self._use_reference:
            self.activate_save_energy_differences(interval=1)
            self.activate_save_displacement(interval=1)

        if logfile is not None:
            self.activate_MDLogger(logfile=logfile, interval=interval_log)
        if interval_save_forces is not None:
            self.activate_save_forces(interval=interval_save_forces)
        if interval_save_trajectory is not None:
            self.activate_save_trajectory(interval=interval_save_trajectory)
        return self

    def set_integrator_Nose_Hoover_NVT(
        self,
        temperature: float = 300.0,
        time_step: float = 1.0,
        ttime: float = 1.0,
        append_trajectory: bool = False,
        initialize: bool = True,
    ):
        """Set integrator using Nose-Hoover NVT thermostat."""
        if initialize:
            self.initialize(temperature)

        self._temperature = temperature
        self._time_step = time_step
        self._thermostat = "Nose-Hoover"
        self._dyn = NPT(
            atoms=self._atoms,
            timestep=time_step * units.fs,
            temperature_K=temperature,
            ttime=ttime * units.fs,
            pfactor=None,
            append_trajectory=append_trajectory,
            # externalstress=1e-07*units.GPa,  # Ignored in NVT
        )
        return self

    def set_integrator_Langevin(
        self,
        temperature: float = 300.0,
        time_step: float = 1.0,
        friction: float = 0.01,
        append_trajectory: bool = False,
        initialize: bool = True,
    ):
        """Set integrator using Langevin dynamics."""
        if initialize:
            self.initialize(temperature)

        self._temperature = temperature
        self._time_step = time_step
        self._thermostat = "Langevin"
        self._dyn = Langevin(
            atoms=self._atoms,
            timestep=time_step * units.fs,
            temperature_K=temperature,
            friction=friction / units.fs,
            append_trajectory=append_trajectory,
        )
        return self

    def initialize(self, temperature: float = 300):
        """Initialize MD."""
        MaxwellBoltzmannDistribution(atoms=self._atoms, temperature_K=temperature)
        Stationary(self._atoms)
        return self

    def run(self, n_eq: int = 5000, n_steps: int = 20000):
        """Run integrator for molecular dynamics."""
        if self._dyn is None:
            raise RuntimeError("Integrator not found.")
        self._n_eq = n_eq
        self._n_steps = n_steps
        self._dyn.run(n_eq + n_steps)
        self._calc_averages(n_eq)
        return self

    def _calc_averages(self, n_eq: int):
        """Calculate averages."""
        if self._energies is not None:
            energies_slice = self._energies[n_eq:]
            self._average_energy = np.average(energies_slice)

            if np.isclose(self._temperature, 0.0):
                self._heat_capacity_eV = 0.0
                self._heat_capacity = 0.0
            else:
                var = np.average(np.square(energies_slice)) - self._average_energy**2
                self._heat_capacity_eV = var / KbEV / self._temperature**2
                self._heat_capacity = (
                    self._heat_capacity_eV * EVtoJ * Avogadro / len(self._atoms.numbers)
                )

        if self._delta_energies is not None:
            self._average_delta_energy = np.average(self._delta_energies[n_eq:])
        if self._displacements is not None:
            self._average_displacement = np.average(self._displacements[n_eq:])
        return self

    @property
    def atoms(self):
        """Return ASE atoms."""
        return self._atoms

    @atoms.setter
    def atoms(self, atoms_in: Atoms):
        """Set ASE atoms."""
        self._atoms = atoms_in
        self._referenced_positions = atoms_in.get_positions()

    @property
    def calculator(self):
        """Return ASE calculator."""
        return self._atoms.calc

    @calculator.setter
    def calculator(self, calc: Calculator):
        """Set ASE calculator."""
        self._atoms.calc = calc

    @property
    def energies(self):
        """Return potential energies in eV/supercell."""
        return np.array(self._energies)

    @property
    def forces(self):
        """Return forces in eV/ang."""
        return np.array(self._forces)

    @property
    def trajectory(self):
        """Return trajectory in ASE atoms."""
        return self._trajectory

    @property
    def delta_energies(self):
        """Return energy differences from reference state in eV/supercell."""
        return np.array(self._delta_energies)

    @property
    def average_energy(self):
        """Return average energy in eV/supercell."""
        return self._average_energy

    @property
    def average_delta_energy(self):
        """Return average energy difference from reference in eV/supercell."""
        return self._average_delta_energy

    @property
    def average_displacement(self):
        """Return average displacement in angstrom."""
        return self._average_displacement

    @property
    def heat_capacity(self):
        """Return heat capacity in J/K/mol."""
        return self._heat_capacity

    @property
    def heat_capacity_eV(self):
        """Return heat capacity in eV."""
        return self._heat_capacity_eV

    def write_conditions(self):
        """Write input conditions as standard output."""
        print("--- Input conditions for MD calculation ---", flush=True)
        print("N atoms:           ", len(self._atoms.numbers), flush=True)
        print("Volume (ang.3):    ", np.round(self._atoms.get_volume(), 5), flush=True)
        print("Thermostat:        ", self._thermostat, flush=True)
        print("Temperature (K):   ", self._temperature, flush=True)
        print("Time step (fs):    ", self._time_step, flush=True)
        if hasattr(self.calculator, "_alpha"):
            print("alpha_fc2:         ", self.calculator._alpha, flush=True)
        print("-------------------------------------------", flush=True)
        return self

    def save_yaml(self, filename="polymlp_md.yaml"):
        """Save properties to yaml file."""
        with open(filename, "w") as f:
            print("system:", self._atoms.symbols, file=f)
            print(file=f)

            print("units:", file=f)
            print("  volume:           angstrom3", file=f)
            print("  temperature:      K", file=f)
            print("  time_step:        fs", file=f)
            print("  average_energy:   eV/supercell", file=f)
            print("  heat_capacity_eV: eV/K/supercell", file=f)
            print("  heat_capacity:    J/K/mol (/Avogadro's number of atoms)", file=f)
            print(file=f)

            print("conditions:", file=f)
            print("  n_atom:     ", len(self._atoms.numbers), file=f)
            print("  volume:     ", self._atoms.get_volume(), file=f)
            print("  thermostat: ", self._thermostat, file=f)
            print("  temperature:", self._temperature, file=f)
            print("  time_step:  ", self._time_step, file=f)
            print("  n_steps_eq: ", self._n_eq, file=f)
            print("  n_steps:    ", self._n_steps, file=f)
            if hasattr(self.calculator, "_alpha"):
                print("  alpha_fc2:  ", self.calculator._alpha, file=f)
            print(file=f)

            print("properties:", file=f)
            print("  average_energy:      ", self._average_energy, file=f)
            print("  heat_capacity_eV:    ", self._heat_capacity_eV, file=f)
            print("  heat_capacity:       ", self._heat_capacity, file=f)
            if self._use_reference:
                print("  average_delta_energy:", self._average_delta_energy, file=f)
