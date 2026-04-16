"""Class for calculating SSCHA properties."""

import numpy as np
from symfc.basis_sets.basis_sets_O1 import FCBasisSetO1

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.api_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.phonopy_utils import phonopy_supercell
from pypolymlp.utils.tensor_utils import compute_tensor_basis_O2


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

        self._proj_force = None
        self._proj_stress = None

    def _get_projector_force(self):
        """Set projector onto symmetrized supercell forces."""
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

    def _get_projector_stress(self):
        """Set projector onto symmetrized stress tensor."""
        proj = compute_tensor_basis_O2(self._sscha_params.unitcell)
        return proj

    def _symmetrize_properties(self, forces: np.ndarray, stress: np.ndarray):
        """Symmetrize forces and stress."""
        if self._proj_force is None:
            raise RuntimeError("Projector of forces not found.")
        if self._proj_stress is None:
            raise RuntimeError("Projector of stress not found.")

        n_supercell = int(round(np.linalg.det(self._sscha_params.supercell_matrix)))
        n_atom_supercell = forces.shape[1]

        forces_sym = (self._proj_force @ forces.T.reshape(-1)).reshape((-1, 3)).T
        unitcell_reps = np.arange(n_atom_supercell) % n_supercell == 0
        forces_sym = forces_sym[:, unitcell_reps]

        order = [0, 3, 5, 3, 1, 4, 5, 4, 2]
        stress_sym = self._proj_stress @ stress[order]
        order = [0, 4, 8, 1, 5, 6]
        stress_sym = stress_sym[order]
        return forces_sym, stress_sym

    def eval(self, structure: PolymlpStructure):
        """Evaluate free energy, forces, and virial stress tensor.

        Properties are composed of SSCHA and static contributions.

        Return
        ------
        free_energy: SSCHA free energy in eV/unitcell.
        force: Forces including static forces in eV/angstrom, shape=(3, n_atom).
        stress: Virial stress tensor in eV/unitcell, shape=(6) for xx, yy, zz, xy, yz, zx.
        """
        # TODO: Implement pressure.
        self._sscha_params.unitcell = structure
        self._proj_force = self._get_projector_force()
        self._proj_stress = self._get_projector_stress()

        self._sscha = run_sscha(self._sscha_params, self._prop, verbose=self._verbose)

        static_energy = self._sscha.properties.static_potential
        sscha_free_energy = self._sscha.properties.free_energy
        free_energy = (static_energy + sscha_free_energy) / EVtoKJmol

        static_forces = self._sscha.properties.static_forces
        average_forces = self._sscha.properties.average_forces
        forces = static_forces + average_forces

        static_stress = self._sscha.properties.static_stress_tensor
        average_stress = self._sscha.properties.average_stress_tensor
        stress = static_stress + average_stress

        forces, stress = self._symmetrize_properties(forces, stress)
        return free_energy, forces, stress

    @property
    def params(self):
        """Parameters of polymlp."""
        return self._prop.params
