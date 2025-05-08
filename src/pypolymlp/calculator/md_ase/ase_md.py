#!/usr/bin/env python

import argparse
from typing import Optional
from typing import Union

import numpy as np

from ase import Atoms, units
from ase.io import read
from ase.md import MDLogger
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
MaxwellBoltzmannDistribution, 
Stationary
)
from ase.calculators.calculator import Calculator

from polymlpase import PolymlpASECalculator


def run_NVT(
    poscar_file: str,
    potentials: list, 
    temperature: int,
    time_step: float,
    ttime: float,
    n_eq: int,
    n_steps: int,
):
    """
    Helper method to run NVT MD simulation.

    Parameters
    ----------
    poscar_file : str
        Path to POSCAR file.
    potentials : list
        List of PolyMLP potential files.
    temperature : int
        Target temperature (K).
    time_step : float
        Time step for MD (fs).
    ttime : float
        Timescale of the thermostat (fs).
    n_eq : int
        Number of equilibration steps.
    n_steps : int
        Number of production steps.
    """
    atoms = read(poscar_file)
    calc = PolymlpASECalculator(potentials=potentials)
    ase_md = MDCalculator_ASE(atoms=atoms, calc=calc)
    ase_md.set_NVT_dynamics(
        temperature=temperature,
        time_step=time_step,
        ttime=ttime,
    )
    ase_md.set_MDLogger(logfile=f"logs_T{temperature}")
    ase_md.run_MD(
        temperature=temperature,
        n_eq=n_eq,
        n_steps=n_steps
    )


def run_Langevin(
    poscar_file: str,
    potentials: list, 
    temperature: int,
    time_step: float,
    friction: float,
    n_eq: int,
    n_steps: int,
):
    """
    Helper method to run Langevin MD simulation.

    Parameters
    ----------
    poscar_file : str
        Path to POSCAR file.
    potentials : list
        List of PolyMLP potential files.
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
    atoms = read(poscar_file)
    calc = PolymlpASECalculator(potentials=potentials)
    ase_md = MDCalculator_ASE(atoms=atoms, calc=calc)
    ase_md.set_Langevin_dynamics(
        temperature=temperature,
        time_step=time_step,
        friction=friction,
    )
    ase_md.set_MDLogger(logfile=f"logs_T{temperature}")
    ase_md.run_MD(
        temperature=temperature,
        n_eq=n_eq,
        n_steps=n_steps
    )


class MDCalculator_ASE:
    """
    MD calculator wrapper for ASE.
    """
    def __init__(
        self,
        atoms: Optional[Atoms] = None,
        calc: Optional[Calculator] = None,
    ):
        """Initialize MDCalculator_ASE."""
        self.atoms = None
        if atoms:
            self.set_atoms(atoms)
        if calc:
            self.set_calculator(calc)

    def set_atoms(self, atoms: Atoms):
        """Set ASEAtoms object."""
        self.atoms = atoms
        self.referenced_positions = atoms.get_positions()

    def set_calculator(self, calc: Calculator):
        """Set ASECalculator."""
        if self.atoms == None:
            raise ValueError("Please set ASEAtoms object.")
        self.atoms.calc = calc

    def set_MDLogger(
        self,
        logfile: str,
        header: bool = True,
        stress: bool = False,
        peratom : bool = True,
        dump_interval: int = 1,
    ):
        """Attach MDLogger to the current dynamics."""
        logger = MDLogger(
            dyn=self._dyn,
            atoms=self.atoms,
            logfile=logfile,
            header=True,
            stress=False,
            peratom=True,
            mode='w',
        )
        self._dyn.attach(logger, interval=dump_interval)

    def set_NVT_dynamics(
        self,
        temperature: int,
        time_step: float,
        ttime: float,
        dump_trajectory: bool = False,
    ):
        """Set up NVT dynamics using Nose-Hoover thermostat."""
        self._dyn = NPT(
            atoms=self.atoms,
            timestep=time_step*units.fs,
            temperature_K=temperature,
            # externalstress=1e-07*units.GPa,  # Ignored in NVT
            ttime=ttime*units.fs,
            pfactor=None,
            append_trajectory=dump_trajectory,
        )

    def set_Langevin_dynamics(
        self,
        temperature: int,
        time_step: float,
        friction: float,
        dump_trajectory: bool = False,
    ):
        """Set up Langevin dynamics."""
        self._dyn = Langevin(
            atoms=self.atoms, 
            timestep=time_step*units.fs,
            temperature_K=temperature,
            friction=friction,
            append_trajectory=dump_trajectory,
        )

    def run_MD(
        self,
        temperature: float,
        n_eq: int,
        n_steps: int,
    ):
        """Run molecular dynamics simulation."""
        MaxwellBoltzmannDistribution(
            atoms=self.atoms,
            temperature_K=temperature
        )
        Stationary(self.atoms)
        self._dyn.run(n_eq+n_steps)
