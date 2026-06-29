"""Class for calculating properties using lammps python API."""

import numpy as np

from pypolymlp.calculator.utils.properties_base import PropertiesBase
from pypolymlp.core.data_format import PolymlpStructure

from .lammps_command import LammpsCommand
from .lammps_structure import convert_structure_to_lammps_format


class PropertiesLammps(PropertiesBase):
    """Class for calculating properties using lammps python API."""

    def __init__(
        self,
        elements: list,
        pot: str = "polymlp.yaml",
        style: str = "polymlp",
        style_command: str = "pair_style",
        coeff_command: str = "pair_coeff",
        log: bool = False,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: potential file.
        """

        super().__init__()
        self._elements = elements
        self._cmd = LammpsCommand(
            elements=elements,
            pot=pot,
            style=style,
            style_command=style_command,
            coeff_command=coeff_command,
            log=log,
            verbose=verbose,
        )

    def eval(self, st: PolymlpStructure):
        """Evaluate properties for a single structure.

        Return
        ------
        energy: Energy. Unit: eV/cell.
        forces: Forces. shape=(3, N), Unit: eV/angstrom.
        stress: Stress tensor.
                shape=(6) in the order of xx, yy, zz, xy, yz, zx. Unit: eV/cell.
        """
        lmp_st = convert_structure_to_lammps_format(st)
        e, f, s = self._cmd.eval(lmp_st)
        f = lmp_st.recover_forces(f)
        s = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
        s = lmp_st.recover_stress(s)
        s = np.array([s[0, 0], s[1, 1], s[2, 2], s[0, 1], s[1, 2], s[2, 0]])
        self._e, self._f, self._s = np.array([e]), [f], np.array([s])
        return e, f, s

    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        e_array, f_array, s_array = [], [], []
        for st in structures:
            e, f, s = self.eval(st)
            e_array.append(e)
            f_array.append(f)
            s_array.append(s)
        self._e = np.array(e)
        self._f = f_array
        self._s = np.array(s)
        return self._e, self._f, self._s

    @property
    def elements(self):
        """Return elements."""
        return self._elements
