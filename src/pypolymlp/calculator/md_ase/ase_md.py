"""Functions for MD integrators in ASE."""

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.md import MDLogger
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


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

    def set_MDLogger(
        self,
        logfile: str,
        header: bool = True,
        stress: bool = False,
        peratom: bool = True,
        mode: str = "w",
        dump_interval: int = 1,
    ):
        """Attach MDLogger to the current dynamics."""
        logger = MDLogger(
            dyn=self._dyn,
            atoms=self._atoms,
            logfile=logfile,
            header=header,
            stress=stress,
            peratom=peratom,
            mode=mode,
        )
        self._dyn.attach(logger, interval=dump_interval)
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

        self._dyn = Langevin(
            atoms=self._atoms,
            timestep=time_step * units.fs,
            temperature_K=temperature,
            friction=friction,
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
        self._dyn.run(n_eq + n_steps)
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
