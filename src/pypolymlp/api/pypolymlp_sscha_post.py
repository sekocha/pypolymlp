"""API class for post-calculations using systematic SSCHA results."""

from typing import Optional

from pypolymlp.calculator.sscha.sscha_properties import SSCHAProperties
from pypolymlp.calculator.sscha.utils.distribution import SSCHADistribution


class PolymlpSSCHAPost:
    """API class for post-calculations using systematic SSCHA results."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._distrib = None

    def compute_thermodynamic_properties(
        self,
        yamlfiles: list[str],
        filename: str = "sscha_properties.yaml",
    ):
        """Calculate thermodynamic properties from SSCHA results."""
        sscha = SSCHAProperties(yamlfiles, verbose=self._verbose)
        sscha.run()
        sscha.save_properties(filename=filename)
        return self

    def init_structure_distribution(
        self,
        yamlfile: str = "sscha_results.yaml",
        fc2file: str = "fc2.hdf5",
        pot: Optional[str] = None,
    ):
        """Load sscha_results.yaml and effective FC2."""
        self._distrib = SSCHADistribution(
            yamlfile=yamlfile,
            fc2file=fc2file,
            pot=pot,
            verbose=self._verbose,
        )
        return self

    def run_structure_distribution(self, n_samples: int = 2000):
        """Calculate properties of structures generated from density matrix."""
        self._distrib.run_structure_distribution(n_samples=n_samples)
        return self

    def save_structure_distribution(self, path="."):
        """Save structures sampled from density matrix and their properties."""
        self._distrib.save_structure_distribution(path=path)
        return self

    @property
    def displacements(self):
        """Return displacements in structures sampled from density matrix.

        shape = (n_supercell, 3, n_atom).
        """
        return self._distrib.displacements

    @property
    def forces(self):
        """Return forces of structures sampled from density matrix.

        shape = (n_supercell, 3, n_atom).
        """
        return self._distrib.forces

    @property
    def energies(self):
        """Return energies of structures sampled from density matrix.

        shape = (n_supercell), unit: eV/supercell.
        """
        return self._distrib.full_potentials

    @property
    def static_potential(self):
        """Return static potential of equilibrium supercell structure.

        Unit: eV/supercell.
        """
        return self._distrib.static_potential

    @property
    def supercells(self):
        """Return supercell structures sampled from density matrix."""
        return self._distrib.supercells
