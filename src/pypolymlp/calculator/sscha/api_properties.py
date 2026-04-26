"""Class for calculating SSCHA properties."""

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.api_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.calculator.sscha.sscha_utils import symmetrize_properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol
from pypolymlp.utils.structure_utils import supercell
from pypolymlp.utils.symfc_utils import compute_projector_cartesian
from pypolymlp.utils.tensor_utils_O2 import compute_projector_O2


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

        self._sscha = None
        self._fc2 = None

    def _get_projector_force(self):
        """Set projector onto symmetrized supercell forces."""
        sup = supercell(
            self._sscha_params.unitcell,
            self._sscha_params.supercell_matrix,
            use_phonopy=True,
        )
        proj = compute_projector_cartesian(sup)
        return proj

    def _get_projector_stress(self):
        """Set projector onto symmetrized stress tensor."""
        proj = compute_projector_O2(self._sscha_params.unitcell)
        return proj

    def _symmetrize_properties(self, forces: np.ndarray, stress: np.ndarray):
        """Symmetrize forces and stress."""
        if self._proj_force is None:
            raise RuntimeError("Projector of forces not found.")
        if self._proj_stress is None:
            raise RuntimeError("Projector of stress not found.")

        forces_sym, stress_sym = symmetrize_properties(
            forces,
            stress,
            self._proj_force,
            self._proj_stress,
            self._sscha_params.n_unitcells,
        )
        return forces_sym, stress_sym

    def eval(self, structure: PolymlpStructure):
        """Evaluate free energy, forces, and virial stress tensor.

        Properties are composed of SSCHA and static contributions.

        Return
        ------
        free_energy: SSCHA free energy in eV/unitcell.
        force: Forces including static forces in eV/angstrom, shape=(3, n_atom).
        stress: Virial stress tensor in eV/unitcell,
                shape=(6) for xx, yy, zz, xy, yz, zx.
        """
        self._sscha_params.unitcell = structure
        self._proj_force = self._get_projector_force()
        self._proj_stress = self._get_projector_stress()

        self._sscha = run_sscha(
            self._sscha_params,
            self._prop,
            verbose=self._verbose,
        )

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

    def eval_multiple(self, structures: list):
        """Evaluate properties of multiple structures.

        Properties are composed of SSCHA and static contributions.

        Return
        ------
        free_energy: List of SSCHA free energy in eV/unitcell.
        force: List of forces including static forces in eV/angstrom, shape=(3, n_atom).
        stress: List of virial stress tensor in eV/unitcell,
                shape=(6) for xx, yy, zz, xy, yz, zx.
        """
        free_energy_all, forces_all, stress_all = [], [], []
        for st in structures:
            free_energy, forces, stress = self.eval(st)
            free_energy_all.append(free_energy)
            forces_all.append(forces)
            stress_all.append(stress)
        return np.array(free_energy_all), forces_all, np.array(stress_all)

    @property
    def params(self):
        """Parameters of polymlp."""
        if self._prop is None:
            return None
        return self._prop.params

    @property
    def properties(self):
        """Return SSCHA results."""
        if self._sscha is None:
            return None
        return self._sscha.properties

    @property
    def logs(self):
        """Return SSCHA progress."""
        if self._sscha is None:
            return None
        return self._sscha.logs

    @property
    def force_constants(self):
        """Force constants at final SSCHA iteration."""
        if self._sscha is None:
            return None
        return self._sscha.force_constants

    @property
    def delta(self):
        """Return convergence delta."""
        if self._sscha is None:
            return None
        return self._sscha.delta
