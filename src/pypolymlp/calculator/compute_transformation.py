"""Class for calculating transformation path."""

import copy
import os
from typing import Optional

import numpy as np

from pypolymlp.calculator.opt_geometry import GeometryOptimization
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.supercell_utils import get_supercell
from pypolymlp.utils.vasp_utils import write_poscar_file


class PolymlpTransformation:
    """Class for calculating transformation pathways."""

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
        self._supercell = structure
        self._supercell_rev = None

        self._axis1, self._axis2, self._other = None, None, None
        self._sd_cell = None
        self._sd_pos = None
        self._energies = None

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
        return self._supercell

    def _change_angle(self, degs: float, axis1: int = 0, axis2: int = 1):
        """Change angle between two axes.

        axis2 is rotated while axis1 and the other axis is fixed.

        [a1, a2, a3].T @ new_a2 = [|a1||a2|cos(degs), a2 @ a2, a2 @ a3].T
        """
        other = ({0, 1, 2} - {axis1, axis2}).pop()
        degs_rad12 = degs * np.pi / 180

        st = self._supercell
        a1, a2, a3 = st.axis[:, axis1], st.axis[:, axis2], st.axis[:, other]
        len1, len2 = np.linalg.norm(a1), np.linalg.norm(a2)
        prod21 = len1 * len2 * np.cos(degs_rad12)
        prod22 = len2 * len2
        prod2other = a2 @ a3
        val = [prod21, prod22, prod2other]

        axisT = np.vstack([a1, a2, a3])
        new_axis = np.linalg.inv(axisT) @ val

        self._supercell_rev = copy.deepcopy(self._supercell)
        self._supercell_rev.axis[:, axis2] = new_axis
        self._axis1, self._axis2, self._other = axis1, axis2, other
        return self._supercell_rev

    def _change_basis(self):
        """Change basis for angle-fixed optimization."""
        basis_a = self._geometry._basis.basis_a
        basis_a_new = np.zeros((basis_a.shape[0], basis_a.shape[1] + 2))
        basis_a_new[:, : basis_a.shape[1]] = basis_a

        axis = self._supercell_rev.axis
        vec = np.zeros((3, 3))
        vec[self._axis1, self._axis1] = 1
        basis_a_new[:, basis_a.shape[1]] = (axis @ vec).reshape(-1)

        vec = np.zeros((3, 3))
        vec[self._other, self._other] = 1
        basis_a_new[:, basis_a.shape[1] + 1] = (axis @ vec).reshape(-1)

        basis_a_new, _ = np.linalg.qr(basis_a_new)
        self._geometry.change_basis_axis(basis_a_new)
        return self._geometry

    def run_single_fix_angle(
        self,
        degs: float,
        axis1: int = 0,
        axis2: int = 1,
        gtol: float = 1e-4,
    ):
        """Run a single geometry optimization with fixed angles."""
        self._change_angle(degs, axis1=axis1, axis2=axis2)
        self._geometry = GeometryOptimization(
            self._supercell_rev,
            self._prop,
            with_sym=True,
            relax_cell=False,
            relax_volume=True,
            relax_positions=True,
            verbose=self._verbose,
        )
        self._change_basis()
        self._geometry.run(gtol=gtol)
        return self._geometry

    def run_fix_angle(
        self,
        degs_min: float,
        degs_max: float,
        degs_int: float = 1,
        axis1: int = 0,
        axis2: int = 1,
        gtol: float = 1e-4,
    ):
        """Run geometry optimizations with fixed angles."""
        self.run_single_fix_angle(degs_min, axis1=axis1, axis2=axis2, gtol=gtol)
        e0 = self.energy

        self._energies = []
        os.makedirs("poscars", exist_ok=True)
        for degs in np.arange(degs_min, degs_max + degs_int, degs_int):
            self.run_single_fix_angle(degs, axis1=axis1, axis2=axis2, gtol=gtol)
            st_conv = self.converged_structure
            e = self.energy
            n = len(st_conv.elements)
            self._energies.append([degs, (e - e0) / n])
            filename = "poscars/POSCAR_deg" + str(degs).zfill(2)
            write_poscar_file(st_conv, filename)

        self._energies = np.array(self._energies)
        return self

    def _shift(self, frac: float, axis_shift: int = 0, axis_normal_shift: int = 1):
        """Provide shift into upper-half cell."""
        self._supercell_rev = copy.deepcopy(self._supercell)
        match = self._supercell.positions[axis_normal_shift] >= 0.5 - 1e-12
        self._supercell_rev.positions[axis_shift, match] += frac

        self._sd_pos = np.ones(self._supercell.positions.shape, dtype=bool)
        self._sd_pos[axis_shift, :] = False
        return self._supercell_rev

    def run_single_fix_shift(
        self,
        frac: float,
        axis_shift: int = 0,
        axis_normal_shift: int = 1,
        gtol: float = 1e-4,
    ):
        """Run a single geometry optimization with fixed shifts."""
        self._supercell_rev = self._shift(frac, axis_shift, axis_normal_shift)
        self._geometry = GeometryOptimization(
            self._supercell_rev,
            self._prop,
            with_sym=False,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
            selective_dynamics_positions=self._sd_pos,
            verbose=self._verbose,
        )
        self._geometry.run(gtol=gtol)
        return self._geometry

    def run_fix_shift(
        self,
        max_shift_frac: float = 0.5,
        n_points: int = 10,
        axis_shift: int = 0,
        axis_normal_shift: int = 1,
        gtol: float = 1e-4,
    ):
        """Run a single geometry optimization with fixed shifts."""
        self.run_single_fix_shift(
            frac=0.0,
            axis_shift=axis_shift,
            axis_normal_shift=axis_normal_shift,
            gtol=gtol,
        )
        e0 = self.energy

        disps = np.arange(0, n_points + 1) * (max_shift_frac / n_points)
        os.makedirs("poscars", exist_ok=True)
        self._energies = []
        for i, disp in enumerate(disps):
            self.run_single_fix_shift(
                disp,
                axis_shift=axis_shift,
                axis_normal_shift=axis_normal_shift,
                gtol=gtol,
            )
            st_conv = self.converged_structure
            e = self.energy
            n = len(st_conv.elements)
            self._energies.append([disp, (e - e0) / n])
            write_poscar_file(st_conv, "poscars/POSCAR_d" + str(i).zfill(2))
        self._energies = np.array(self._energies)
        return self

    def save(self, filename: str = "polymlp_path_energy.dat"):
        """Save energies along path."""
        header = "disp., energy (eV/atom)"
        np.savetxt(filename, self._energies, fmt="%f", header=header)

    @property
    def energies(self):
        """Return energies along path."""
        return self._energies

    @property
    def energy(self):
        """Return energy."""
        if self._geometry is None:
            return None
        return self._geometry.energy

    @property
    def success(self):
        """Return whether optimization is successed."""
        if self._geometry is None:
            return None
        return self._geometry.success

    @property
    def converged_structure(self):
        """Return converged structure."""
        if self._geometry is None:
            return None
        return self._geometry.structure
