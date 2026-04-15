"""Class for calculating SSCHA properties."""

import numpy as np
from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.api_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.phonopy_utils import phonopy_supercell


class PropertiesSSCHA:
    """Class for calculating SSCHA properties."""

    def __init__(
        self,
        sscha_params: SSCHAParams,
        properties: Properties,
        precondition: bool = True,
        use_temporal_cutoff: bool = False,
        path: str = "./sscha",
        write_pdos: bool = False,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        sscha_params: SSCHA parameters in SSCHAParams class.
        properties: Properties instance.
        """
        self._sscha_params = sscha_params
        self._prop = properties
        self._verbose = verbose

        if self._sscha_params.supercell_matrix is None:
            self._sscha_params.supercell_matrix = np.eye(3, dtype=int)

    def _get_projector(self):
        """Set projector of supercell forces onto unitcell forces."""
        supercell = phonopy_supercell(
            self._sscha_params.unitcell,
            supercell_matrix=self._sscha_params.supercell_matrix,
        )
        try:
            basis = FCBasisSetO1(supercell, use_mkl=False).run()
            basis_matrix = basis.full_basis_set.toarray()
            if len(basis_matrix) == 0:
                three_n = len(supercell.symbols) * 3
                return np.zeros((three_n, three_n))
            return basis_matrix @ basis_matrix.T
        except:
            three_n = len(supercell.symbols) * 3
            return np.zeros((three_n, three_n))

    def eval(self, structure: PolymlpStructure):
        """Evaluate free energy, forces, and virial stress tensor.

        Properties are composed of SSCHA and static contributions.

        Return
        ------
        free_energy: SSCHA free energy in eV/unitcell.
        force: Forces including static forces in eV/angstrom, shape=(3, n_atom).
        stress: Virial stress tensor in eV/unitcell, shape=(6) for xx, yy, zz, xy, yz, zx.
        """
        self._sscha_params.unitcell = structure
        proj = self._get_projector()

        self._sscha = run_sscha(self._sscha_params, self._prop, verbose=self._verbose)

        static_energy = self._sscha.properties.static_potential
        sscha_free_energy = self._sscha.properties.free_energy
        free_energy = (static_energy + sscha_free_energy) / EVtoKJmol

        static_forces = self._sscha.properties.static_forces
        average_forces = self._sscha.properties.average_forces
        forces = static_forces + average_forces

        n_supercell = int(round(np.linalg.det(self._sscha_params.supercell_matrix)))
        n_atom_supercell = forces.shape[1]
        forces = (proj @ forces.T.reshape(-1)).reshape((-1, 3)).T
        unitcell_reps = np.arange(n_atom_supercell) % n_supercell == 0
        forces = forces[:, unitcell_reps]

        static_stress = self._sscha.properties.static_stress_tensor
        average_stress = self._sscha.properties.average_stress_tensor
        stress = static_stress + average_stress
        return free_energy, forces, stress

    @property
    def params(self):
        """Parameters of polymlp."""
        return self._prop.params
