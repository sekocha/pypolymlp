"""Class for calculating generalized stacking fault energies."""

import copy
import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.supercell_utils import get_supercell
from pypolymlp.utils.vasp_utils import write_poscar_file

eVang2ToJm2 = 16.021766343


class PolymlpGSFE:
    """Class for calculating generalized stacking fault energies."""

    def __init__(
        self,
        structure: PolymlpStructure,
        properties: Properties,
        verbose: bool = False,
    ):
        """Init method."""
        self._prop = properties
        self._verbose = verbose

        self._base_structure = structure
        self._supercell = None
        self._supercell_disp = None
        self._area = None

        self._sd_cell = None
        self._sd_pos = None
        self._shift_atoms = None
        self._excess_energies = None

    def set_supercell(
        self,
        disp1: tuple = (1, 0, 0),
        disp2: tuple = (0, 1, 0),
        slip_plane: tuple = (0, 0, 1),
        n_layers: int = 2,
        supercell_matrix: Optional[np.ndarray] = None,
    ):
        """Set supercell."""
        if supercell_matrix is not None:
            if np.array(supercell_matrix).shape != (3, 3):
                raise RuntimeError("Supercell matrix shape is not (3, 3).")
            matrix = copy.deepcopy(supercell_matrix)
        else:
            if len(disp1) != 3:
                raise RuntimeError("Three elements required for disp1.")
            if len(disp2) != 3:
                raise RuntimeError("Three elements required for disp2.")
            if len(slip_plane) != 3:
                raise RuntimeError("Three elements required for slip plane.")
            matrix = np.zeros((3, 3), dtype=int)
            matrix[:, 0] = np.array(disp1)
            matrix[:, 1] = np.array(disp2)
            matrix[:, 2] = np.array(slip_plane) * n_layers

        for i in range(3):
            if matrix[i, i] < 0:
                matrix[:, i] *= -1

        self._supercell = get_supercell(self._base_structure, matrix)
        self._area = np.linalg.norm(self._supercell.axis[:, 0]) * np.linalg.norm(
            self._supercell.axis[:, 1]
        )

        self._sd_pos = np.ones(self._supercell.positions.shape, dtype=bool)
        self._sd_pos[0, :] = False
        self._sd_pos[1, :] = False
        self._sd_cell = np.zeros((3, 3), dtype=bool)
        self._sd_cell[:, 2] = True
        self._shift_atoms = self._supercell.positions[2] >= 0.5 - 1e-12
        return self._supercell

    def _change_structure(self, disp1: float, disp2: float):
        """Introduce displacement into supercell."""
        if self._supercell is None:
            raise RuntimeError("Supercell not found.")

        self._supercell_disp = copy.deepcopy(self._supercell)
        self._supercell_disp.positions[0, self._shift_atoms] += disp1
        self._supercell_disp.positions[1, self._shift_atoms] += disp2
        return self._supercell_disp

    def run_single(self, disp1: float, disp2: float, gtol: float = 1e-4):
        """Run geometry optimization for single displaced structure."""
        self._supercell_disp = self._change_structure(disp1, disp2)
        self._geometry = GeometryOptimization(
            self._supercell_disp,
            self._prop,
            with_sym=False,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
            selective_dynamics_cell=self._sd_cell,
            selective_dynamics_positions=self._sd_pos,
            verbose=False,
        ).run(gtol=gtol)
        return self._geometry

    def run(self, n_points: int = 10, gtol: float = 1e-4):
        """Run geometry optimizations for entire set of displaced structures."""
        go = self.run_single(0.0, 0.0)
        e0 = go.energy

        disps1 = np.arange(n_points + 1) * (0.5 / n_points)
        os.makedirs("poscars", exist_ok=True)
        self._excess_energies = []
        for i, d1 in enumerate(disps1):
            if self._verbose:
                print("Displacement along axis 1:", np.round(d1, 3), flush=True)
            for j, d2 in enumerate(disps1):
                go = self.run_single(d1, d2, gtol=gtol)
                if not go.success:
                    continue

                excess_e = (go.energy - e0) / self._area / 2
                excess_e_Jm2 = excess_e * eVang2ToJm2
                filename = "poscars/POSCAR_" + str(i).zfill(2) + "_" + str(j).zfill(2)
                write_poscar_file(go.structure, filename)
                self._excess_energies.append([d1, d2, excess_e_Jm2])
        self._excess_energies = np.array(self._excess_energies)
        return self

    def save(self, filename: str = "polymlp_gsfe.dat"):
        """Save excess energies."""
        header = "1st disp., 2nd disp, Delta E"
        np.savetxt(filename, self._excess_energies, fmt="%f", header=header)

    @property
    def excess_energies(self):
        """Return excess energies for generalized stacking fault energies.

        Return
        ------
        excess_energies: Excess energies in J/m^2, shape=(n_points**2, 3).
                      (disp. along 1st axis, disp. along 2nd axis, excess eneergy).
        """
        return self._excess_energies
