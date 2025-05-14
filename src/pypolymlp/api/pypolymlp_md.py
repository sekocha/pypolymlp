"""API Class for performing MD simulations."""

from typing import Literal, Optional, Union

import numpy as np
from ase.calculators.calculator import Calculator

from pypolymlp.calculator.md.ase_md import IntegratorASE
from pypolymlp.calculator.md.md_utils import (
    calc_integral,
    get_p_roots,
    save_thermodynamic_integration_yaml,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.utils.ase_calculator import (
    PolymlpASECalculator,
    PolymlpFC2ASECalculator,
)
from pypolymlp.calculator.utils.ase_utils import (
    ase_atoms_to_structure,
    structure_to_ase_atoms,
)
from pypolymlp.calculator.utils.fc_utils import load_fc2_hdf5
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal


# TODO: Implement Nose-Hoover-chain thermostat.
class PypolymlpMD:
    """API Class for performing MD simulations."""

    def __init__(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties instance.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        self._calculator = PolymlpASECalculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            require_mlp=False,
        )
        self._verbose = verbose

        self._unitcell = None
        self._supercell = None
        self._unitcell_ase = None
        self._supercell_ase = None
        self._integrator = None

        self._use_reference = False
        self._delta_energies = None
        self._delta_free_energy = None

    def set_ase_calculator(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
    ):
        """Set ASE calculator with polymlp."""
        self._calculator = PolymlpASECalculator(
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
        )
        return self

    def set_ase_calculator_with_fc2(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
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
        if self._supercell is None:
            raise RuntimeError("Supercell not found.")

        fc2 = load_fc2_hdf5(fc2hdf5, return_matrix=True)
        assert fc2.shape[0] == self._supercell.positions.shape[1] * 3

        self._calculator = PolymlpFC2ASECalculator(
            fc2,
            self._supercell,
            pot=pot,
            params=params,
            coeffs=coeffs,
            properties=properties,
            alpha=alpha,
        )
        self._use_reference = True
        return self

    def load_poscar(self, poscar: str):
        """Parse POSCAR file and supercell matrix."""
        self._unitcell = Poscar(poscar).structure
        self._unitcell_ase = structure_to_ase_atoms(self._unitcell)
        self._supercell = self._unitcell
        self._supercell_ase = self._unitcell_ase
        return self

    def set_supercell(self, size: tuple):
        """Set supercell from unitcell."""
        if self._unitcell is None:
            raise RuntimeError("Unitcell not found.")
        if len(size) != 3:
            raise RuntimeError("Supercell size is not equal to 3.")
        self._supercell = supercell_diagonal(self._unitcell, size)
        self._supercell_ase = structure_to_ase_atoms(self._supercell)
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
            self._integrator.activate_standard_output(interval=100)

        self._integrator.run(n_eq=n_eq, n_steps=n_steps)
        return self

    def run_thermodynamic_integration(
        self,
        thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
        n_alphas: int = 15,
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
        if not self._use_reference:
            raise RuntimeError("Reference state not found in Calculator.")

        alphas, weights = get_p_roots(n=n_alphas, a=0.0, b=1.0)
        self._delta_energies = []
        for alpha in alphas:
            if self._verbose:
                print("TI (alpha):", alpha, flush=True)

            self._calculator.alpha = alpha
            if thermostat == "Nose-Hoover":
                self.run_Nose_Hoover_NVT(
                    temperature=temperature,
                    time_step=time_step,
                    ttime=ttime,
                    n_eq=n_eq,
                    n_steps=n_steps,
                    interval_log=None,
                    logfile=None,
                )
            elif thermostat == "Langevin":
                self.run_Langevin(
                    temperature=temperature,
                    time_step=time_step,
                    friction=friction,
                    n_eq=n_eq,
                    n_steps=n_steps,
                    interval_log=None,
                    logfile=None,
                )
            self._delta_energies.append([alpha, self.average_delta_energy])
        self._delta_energies = np.array(self._delta_energies)
        self._delta_free_energy = calc_integral(
            weights,
            self._delta_energies[:, 1],
            a=0.0,
            b=1.0,
        )
        return self

    def _check_requisites(self):
        """Check requisites for MD simulations."""
        if self._supercell_ase is None:
            raise RuntimeError("Supercell not found.")
        if self._calculator is None:
            raise RuntimeError("Calculator not found.")

    def save_yaml(self, filename: str = "polymlp_md.yaml"):
        """Save properties to yaml file."""
        self._integrator.save_yaml(filename=filename)
        return self

    def save_thermodynamic_integration_yaml(self, filename: str = "polymlp_ti.yaml"):
        """Save results of thermodynamic integration."""
        if not self._use_reference:
            raise RuntimeError("Reference state not found in Calculator.")

        save_thermodynamic_integration_yaml(
            self._integrator,
            self._delta_free_energy,
            self._delta_energies,
            filename=filename,
        )
        return self

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
    def calculator(self):
        """Return calculator."""
        return self._calculator

    @calculator.setter
    def calculator(self, calc: Calculator):
        """Set calculator."""
        self._calculator = calc

    @property
    def energies(self):
        """Return potential energies."""
        return self._integrator.energies

    @property
    def forces(self):
        """Return forces."""
        return self._integrator.forces

    @property
    def trajectory(self):
        """Return trajectory."""
        return [ase_atoms_to_structure(t) for t in self._integrator.trajectory]

    @property
    def delta_energies(self):
        """Return potential energies from reference state."""
        return self._integrator.delta_energies

    @property
    def average_energy(self):
        """Return avarage energy."""
        return self._integrator.average_energy

    @property
    def average_delta_energy(self):
        """Return avarage energy."""
        return self._integrator.average_delta_energy

    @property
    def heat_capacity(self):
        """Return heat capacity."""
        return self._integrator.heat_capacity

    @property
    def final_structure(self):
        """Return structure at the final step."""
        return ase_atoms_to_structure(self._supercell_ase)


def run_thermodynamic_integration(
    pot: str = "polymlp.yaml",
    poscar: str = "POSCAR",
    supercell_size: tuple = (1, 1, 1),
    thermostat: Literal["Nose-Hoover", "Langevin"] = "Langevin",
    n_alphas: int = 15,
    fc2hdf5: str = "fc2.hdf5",
    temperature: float = 300.0,
    time_step: float = 1.0,
    ttime: float = 20.0,
    friction: float = 0.01,
    n_eq: int = 2000,
    n_steps: int = 20000,
    filename: str = "polymlp_ti.yaml",
    verbose: bool = True,
):
    """Run thermodynamic integration.

    Parameters
    ----------
    pot: polymlp file.
    poscar: Structure in POSCAR format.
    supercell_size: Diagonal supercell size.
    thermostat: Thermostat.
    n_alphas: Number of sample points for thermodynamic integration
              using Gaussian quadrature.
    fc2hdf5: HDF5 file for second-order force constants.
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

    md = PypolymlpMD(verbose=verbose)
    md.load_poscar(poscar)
    md.set_supercell(supercell_size)
    md.set_ase_calculator_with_fc2(pot=pot, fc2hdf5=fc2hdf5, alpha=0.0)
    md.run_thermodynamic_integration(
        thermostat=thermostat,
        temperature=temperature,
        time_step=time_step,
        ttime=ttime,
        friction=friction,
        n_eq=n_eq,
        n_steps=n_steps,
    )
    md.save_thermodynamic_integration_yaml(filename=filename)
    return md
