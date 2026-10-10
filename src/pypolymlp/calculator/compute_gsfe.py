"""Class for calculating generalized stacking fault energies."""

import copy
import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.supercell_utils import get_supercell_three_directions
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
        self._null_space_basis = None

        self._excess_energies = None

    def set_supercell(
        self,
        disp1: tuple = (1, 0, 0),
        disp2: tuple = (0, 1, 0),
        glide_plane: tuple = (0, 0, 1),
        n_layers: int = 2,
        supercell_matrix: Optional[np.ndarray] = None,
    ):
        """Set supercell."""
        self._supercell = get_supercell_three_directions(
            self._base_structure,
            direction1=disp1,
            direction2=disp2,
            direction3=glide_plane,
            n_layers=n_layers,
            supercell_matrix=supercell_matrix,
        )
        self._set_supercell_params(disp1=disp1, disp2=disp2)

        if hasattr(self._prop, "change_unit_cell"):
            self._prop.change_unit_cell(self._supercell)
        if hasattr(self._prop, "set_null_space_basis"):
            self._prop.set_null_space_basis(self._null_space_basis)

        return self._supercell

    def _set_supercell_params(
        self,
        disp1: tuple = (1, 0, 0),
        disp2: tuple = (0, 1, 0),
    ):
        """Set parameters for supercell."""
        if self._supercell is None:
            raise RuntimeError("Supercell not found.")

        len0 = np.linalg.norm(self._supercell.axis[:, 0])
        len1 = np.linalg.norm(self._supercell.axis[:, 1])
        self._area = len0 * len1

        self._sd_pos = np.ones(self._supercell.positions.shape, dtype=bool)
        self._sd_pos[0, :] = False
        self._sd_pos[1, :] = False
        self._sd_cell = np.zeros((3, 3), dtype=bool)
        self._sd_cell[:, 2] = True
        self._shift_atoms = self._supercell.positions[2] >= 0.5 - 1e-12

        shape = (self._sd_pos.shape[0], self._sd_pos.shape[1], 2)
        null_space_basis = np.zeros(shape)
        # null_space_basis[:, self._shift_atoms, 0] = np.array(disp1)[:, None]
        # null_space_basis[:, self._shift_atoms, 1] = np.array(disp2)[:, None]

        # TODO: Correct in non-orthogonal system?
        null_space_basis[0, self._shift_atoms, 0] = 1.0
        null_space_basis[1, self._shift_atoms, 1] = 1.0

        for i in range(2):
            ave = np.sum(null_space_basis[:, :, i], axis=1) / null_space_basis.shape[1]
            null_space_basis[:, :, i] -= ave[:, None]

        null_space_basis = null_space_basis.transpose((1, 0, 2)).reshape((-1, 2))
        self._null_space_basis, _ = np.linalg.qr(null_space_basis)
        return self

    def _change_structure(self, disp1: float, disp2: float):
        """Introduce displacement into supercell."""
        if self._supercell is None:
            raise RuntimeError("Supercell not found.")

        self._supercell_disp = copy.deepcopy(self._supercell)
        self._supercell_disp.positions[0, self._shift_atoms] += disp1
        self._supercell_disp.positions[1, self._shift_atoms] += disp2
        return self._supercell_disp

    def run_single(
        self,
        disp1: float = 0.0,
        disp2: float = 0.0,
        gtol: float = 1e-4,
        maxiter: int = 1000,
        restart: bool = False,
    ):
        """Run geometry optimization for single displaced structure.

        Parameters
        ----------
        disp1: Shift magnitude for first direction in fractional coordinates.
        disp2: Shift magnitude for second direction in fractional coordinates.
        gtol: Tolerance for gradients in geometry optimization.

        Return
        ------
        Energy: Energy per stacking fault in J/m^2.
                Excess stacking fault energy can be calculated as
                energy - energy(disp1=0.0, disp2=0.0).
        """
        if restart:
            self._supercell = self._base_structure
            self._supercell_disp = self._base_structure
            self._set_supercell_params()
        else:
            self._supercell_disp = self._change_structure(disp1, disp2)

        if hasattr(self._prop, "_prop"):
            go = GeometryOptimization(
                self._supercell_disp,
                self._prop._prop,
                with_sym=False,
                relax_cell=True,
                relax_volume=True,
                relax_positions=True,
                selective_dynamics_cell=self._sd_cell,
                selective_dynamics_positions=self._sd_pos,
                verbose=self._verbose,
            )

            # go.change_basis_axis(go._basis_a[:, 2:])
            go.run(gtol=1e-4, maxiter=10000)
            if go.success:
                self._supercell_disp = go.structure

        go = GeometryOptimization(
            self._supercell_disp,
            self._prop,
            with_sym=False,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
            selective_dynamics_cell=self._sd_cell,
            selective_dynamics_positions=self._sd_pos,
            verbose=self._verbose,
        )
        go.change_basis_axis(go._basis_a[:, 2:])
        go.run(gtol=gtol, maxiter=maxiter)

        try:
            energy = go.energy / self._area / 2
            energy_Jm2 = energy * eVang2ToJm2
            return (energy_Jm2, go)
        except:
            return (None, go)

    def run(self, n_points: int = 10, gtol: float = 1e-4, maxiter: int = 1000):
        """Run geometry optimizations for entire set of displaced structures."""
        e0, _ = self.run_single(0.0, 0.0)
        if e0 is None:
            raise RuntimeError("Calculation for perfect crystal failed.")

        disps1 = np.arange(n_points + 1) * (0.5 / n_points)
        os.makedirs("poscars", exist_ok=True)
        self._excess_energies = []
        for i, d1 in enumerate(disps1):
            if self._verbose:
                print("Displacement along axis 1:", np.round(d1, 3), flush=True)
            for j, d2 in enumerate(disps1):
                e_disp, go = self.run_single(d1, d2, gtol=gtol, maxiter=maxiter)
                if e_disp is None:
                    continue

                excess_e_Jm2 = e_disp - e0
                filename = "poscars/POSCAR_" + str(i).zfill(2) + "_" + str(j).zfill(2)
                write_poscar_file(go.structure, filename)
                self._excess_energies.append([d1, d2, excess_e_Jm2])
        self._excess_energies = np.array(self._excess_energies)
        return self._excess_energies

    def save(self, filename: str = "polymlp_gsfe.dat"):
        """Save excess energies."""
        header = "1st disp., 2nd disp, Delta E (J/m2)"
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
