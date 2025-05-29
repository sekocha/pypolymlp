"""API class for post-calculations using systematic SSCHA results."""

from typing import Optional

from pypolymlp.calculator.sscha.utils.distribution import SSCHADistribution


class PypolymlpSSCHAPost:
    """API class for post-calculations using systematic SSCHA results."""

    def __init__(self, verbose: bool = False):
        """Init method."""
        self._verbose = verbose
        self._distrib = None

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

    def save_structure_distribution(self, path: str = ".", save_poscars: bool = False):
        """Save structures sampled from density matrix and their properties."""
        self._distrib.save_structure_distribution(path=path, save_poscars=save_poscars)
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


#     def compute_thermodynamic_properties(
#         self,
#         yamlfiles: list[str],
#         filename: str = "sscha_properties.yaml",
#     ):
#         """Calculate thermodynamic properties from SSCHA results."""
#         sscha = SSCHAProperties(yamlfiles, verbose=self._verbose)
#         sscha.run()
#         sscha.save_properties(filename=filename)
#         sscha.save_equilibrium_structures(path="sscha_eqm_poscars")
#         return self
#
#     def find_phase_transition(self, yaml1: str, yaml2: str):
#         """Find phase transition and its temperature.
#
#         Parameters
#         ----------
#         yaml1: sscha_properties.yaml for the first structure.
#         yaml2: sscha_properties.yaml for the second structure.
#         """
#         tc_linear, tc_quartic = find_transition(yaml1, yaml2)
#         return tc_linear, tc_quartic
#
#     def compute_phase_boundary(self, yaml1: str, yaml2: str):
#         """Compute phase boundary between two structures.
#
#         Parameters
#         ----------
#         yaml1: sscha_properties.yaml for the first structure.
#         yaml2: sscha_properties.yaml for the second structure.
#
#         Return
#         ------
#         boundary: [pressures, temperatures].
#         """
#         boundary = compute_phase_boundary(yaml1, yaml2)
#         return boundary
