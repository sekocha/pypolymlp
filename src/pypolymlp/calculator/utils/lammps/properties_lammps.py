"""Class for calculating properties using lammps python API."""

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
        self._e, self._f, self._s = [e], [f], [s]
        return e, f, s

    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        pass
        # self._e, self._f, self._s = self._prop.eval_multiple(structures)
        # self._structures = structures
        # return self._e, self._f, self._s

    @property
    def elements(self):
        """Return elements."""
        return self._elements
