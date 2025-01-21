"""Class for generating structures from force constants and calculating their properties."""

import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_utils import Restart
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
        self._verbose = verbose
        self._res = Restart(yamlfile, fc2hdf5=fc2file)
        pot = self._res.polymlp if pot is None else pot
        prop = Properties(pot=pot)
        self._ph_real = HarmonicReal(
            self._res.supercell,
            prop,
            n_unitcells=self._res.n_unitcells,
            fc2=self._res.force_constants,
        )

        if self._verbose:
            print("Load SSCHA results:")
            print("  yaml:        ", yamlfile)
            print("  fc2 :        ", fc2file)
            print("  mlp :        ", pot)
            print("  temperature :", self._res.temperature)

    def run_structure_distribution(self, n_samples: int = 2000):
        """Calculate properties of structures generated from density matrix."""
        self._ph_real.run(t=self._res.temperature, n_samples=n_samples)
        return self

    def save_structure_distribution(self, path: str = ".", save_poscars: bool = False):
        """Save structures sampled from density matrix and their properties."""
        disps = self.displacements.transpose((0, 2, 1))
        forces = self.forces.transpose((0, 2, 1))
        np.save(path + "/sscha_disps.npy", disps)
        np.save(path + "/sscha_forces.npy", forces)
        np.save(path + "/sscha_energies.npy", np.array(self.energies))
        with open(path + "/sscha_averages.yaml", "w") as f:
            save_cell(self._res.unitcell, tag="unitcell", file=f)
            print("supercell_matrix:", file=f)
            print(" -", list(self._res.supercell_matrix[0].astype(int)), file=f)
            print(" -", list(self._res.supercell_matrix[1].astype(int)), file=f)
            print(" -", list(self._res.supercell_matrix[2].astype(int)), file=f)
            print("", file=f)
            print("static_potential: ", self.static_potential, file=f)
            print("", file=f)
            print("average_potential:", np.mean(self.energies), file=f)
            print("", file=f)
            # np.set_printoptions(suppress=True)
            print_array2d(self.average_forces.T, "average_forces", f, indent_l=0)

        if save_poscars:
            if self._verbose:
                print("Save POSCAR files.", flush=True)
            os.makedirs(path + "/sscha_poscars", exist_ok=True)
            for i, st in enumerate(self.supercells, 1):
                filename = path + "/sscha_poscars/POSCAR-" + str(i).zfill(4)
                write_poscar_file(st, filename=filename)
            print("sscha_poscars/POSCAR* are generated.", flush=True)

        if self._verbose:
            print("sscha_disps.npy and sscha_forces.npy are generated.", flush=True)
            print("- shape:", forces.shape, flush=True)
            print("Potential energies of supercells are generated.", flush=True)
            print("- shape:", len(self.energies), flush=True)
            print(
                "- static_potential: ",
                self.static_potential,
                "(eV/supercell)",
                flush=True,
            )
            print(
                "- average_potential:",
                np.mean(self.energies),
                "(eV/supercell)",
                flush=True,
            )
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
