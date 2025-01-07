"""Class for generating structures from force constants and calculating their properties."""

import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_utils import Restart
from pypolymlp.utils.vasp_utils import write_poscar_file
from pypolymlp.utils.yaml_utils import save_cell


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

    def save_structure_distribution(self, path="."):
        """Save structures sampled from density matrix and their properties."""
        disps = self.displacements.transpose((0, 2, 1))
        forces = self.forces.transpose((0, 2, 1))
        np.save(path + "/sscha_disps.npy", disps)
        np.save(path + "/sscha_forces.npy", forces)
        np.save(path + "/sscha_energies.npy", np.array(self.energies))
        with open(path + "/sscha_potentials.yaml", "w") as f:
            save_cell(self._res.unitcell, tag="unitcell", file=f)
            print("supercell_matrix:", file=f)
            print(" -", list(self._res.supercell_matrix[0].astype(int)), file=f)
            print(" -", list(self._res.supercell_matrix[1].astype(int)), file=f)
            print(" -", list(self._res.supercell_matrix[2].astype(int)), file=f)
            print("", file=f)
            print("static_potential: ", self.static_potential, file=f)
            print("average_potential:", np.mean(self.energies), file=f)

        os.makedirs(path + "/sscha_poscars", exist_ok=True)
        for i, st in enumerate(self.supercells, 1):
            filename = path + "/sscha_poscars/POSCAR-" + str(i).zfill(4)
            write_poscar_file(st, filename=filename)

        if self._verbose:
            print("sscha_disps.npy and sscha_forces.npy are generated.", flush=True)
            print("- shape:", forces.shape, flush=True)
            print("Potential energies of supercells are generated.")
            print("- shape:", len(self.energies))
            print("- static_potential: ", self.static_potential, "(eV/supercell)")
            print("- average_potential:", np.mean(self.energies), "(eV/supercell)")
            print("sscha_poscars/POSCAR* are generated.")
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
    def supercells(self):
        """Return supercell structures sampled from density matrix."""
        return self._ph_real.supercells
