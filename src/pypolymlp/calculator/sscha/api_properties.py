"""Class for calculating SSCHA properties."""

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.api_sscha import run_sscha
from pypolymlp.calculator.sscha.sscha_params import SSCHAParams
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.units import EVtoKJmol


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

    def eval(self, structure: PolymlpStructure):
        """Evaluate free energy, forces, and virial stress tensor.

        Properties are composed of SSCHA and static contributions.

        Return
        ------
        free_energy: SSCHA free energy in eV/cell.
        force: Forces including static forces in eV/angstrom, shape=(3, n_atom).
        stress: Virial stress tensor in eV/cell, shape=(6) for xx, yy, zz, xy, yz, zx.
        """
        self._sscha_params.unitcell = structure
        self._sscha_params.supercell_matrix = np.eye(3, dtype=int)
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
        return free_energy, forces, stress

    @property
    def params(self):
        """Parameters of polymlp."""
        return self._prop.params
