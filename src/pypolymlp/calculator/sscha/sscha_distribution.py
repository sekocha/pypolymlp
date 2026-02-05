"""Class for generating structures from force constants and calculating their properties."""

import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_restart import Restart
from pypolymlp.utils.vasp_utils import write_poscar_file
from pypolymlp.utils.yaml_utils import print_array2d, save_cell


class SSCHADistribution:
    """Class for generating structures from FC2 and calculating their properties."""

    def __init__(
        self,
        yamlfile: str = "sscha_results.yaml",
        fc2file: str = "fc2.hdf5",
        pot: Optional[str] = None,
        verbose: bool = False,
    ):
        """Init method.

        Load sscha_results.yaml and effective FC2.
        """
        self._res = Restart(yamlfile, fc2hdf5=fc2file, pot=pot)
        prop = Properties(pot=self._res.polymlp)
        self._ph_real = HarmonicReal(
            self._res.supercell,
            prop,
            n_unitcells=self._res.n_unitcells,
            fc2=self._res.force_constants,
            verbose=verbose,
        )
        self._verbose = verbose

        if self._verbose:
            print("Load SSCHA results:")
            print("  yaml:        ", yamlfile)
            print("  fc2 :        ", fc2file)
            print("  polymlp :    ", pot)
            print("  temperature :", self._res.temperature)

    def run_structure_distribution(self, n_samples: int = 2000):
        """Calculate properties of structures generated from density matrix."""
        self._ph_real.run(temp=self._res.temperature, n_samples=n_samples)
        return self

    def save_structure_distribution(self, path: str = ".", save_poscars: bool = False):
        """Save structures sampled from density matrix and their properties."""
        os.makedirs(path, exist_ok=True)
        disps = self.displacements.transpose((0, 2, 1))
        forces = self.forces.transpose((0, 2, 1))
        np.save(path + "/sscha_disps.npy", disps)
        np.save(path + "/sscha_forces.npy", forces)
        np.save(path + "/sscha_energies.npy", np.array(self.energies))
        self._save_averages(path)

        if save_poscars:
            self._save_poscars(path)
        if self._verbose:
            self._print_summary()
        return self

    def _save_averages(self, path: str = "."):
        """Save average properties."""
        with open(path + "/sscha_averages.yaml", "w") as f:
            save_cell(self._res.unitcell, tag="unitcell", file=f)
            intmat = self._res.supercell_matrix.astype(int)
            print_array2d(intmat, "supercell_matrix", f, indent_l=0)
            print(file=f)
            print("static_potential: ", self.static_potential, file=f)
            print(file=f)
            print("average_potential:", np.mean(self.energies), file=f)
            print(file=f)
            print_array2d(self.average_forces.T, "average_forces", f, indent_l=0)
        return self

    def _print_summary(self):
        """Print summary of saving distribution."""
        print("sscha_disps.npy and sscha_forces.npy are generated.", flush=True)
        print("- shape:", self.forces.transpose((0, 2, 1)).shape, flush=True)
        print("Potential energies of supercells are generated.", flush=True)
        print("- shape:", len(self.energies), flush=True)
        unit = "(eV/supercell)"
        print("- static_potential: ", self.static_potential, unit, flush=True)
        print("- average_potential:", np.mean(self.energies), unit, flush=True)
        return self

    def _save_poscars(self, path: str = "."):
        """Save POSCAR files."""
        if self._verbose:
            print("Save POSCAR files.", flush=True)

        path_poscars = path + "/sscha_poscars"
        os.makedirs(path_poscars, exist_ok=True)
        for i, st in enumerate(self.supercells, 1):
            filename = path_poscars + "/POSCAR-" + str(i).zfill(4)
            write_poscar_file(st, filename=filename)
        if self._verbose:
            print("POSCARs are generated in", path_poscars, flush=True)
        return self

    @property
    def displacements(self):
        """Return displacements in structures sampled from density matrix.

        shape = (n_supercell, 3, n_atom).
        """
        return self._ph_real.displacements

    @property
    def forces(self):
        """Return forces of structures sampled from density matrix.

        shape = (n_supercell, 3, n_atom).
        """
        return self._ph_real.forces

    @property
    def energies(self):
        """Return energies of structures sampled from density matrix.

        shape = (n_supercell), unit: eV/supercell.
        """
        return self._ph_real.full_potentials

    @property
    def static_potential(self):
        """Return static potential of equilibrium supercell structure.

        Unit: eV/supercell.
        """
        return self._ph_real.static_potential

    @property
    def static_forces(self):
        """Return static forces of equilibrium supercell structure.

        Unit: eV/angstrom.
        """
        return self._ph_real.static_forces

    @property
    def average_forces(self):
        """Return average forces.

        Unit: eV/angstrom.
        """
        return self._ph_real.average_forces

    @property
    def supercells(self):
        """Return supercell structures sampled from density matrix."""
        return self._ph_real.supercells
