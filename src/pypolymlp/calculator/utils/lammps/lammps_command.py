"""Class for controlling python lammps object."""

from typing import Optional

import numpy as np
from lammps import lammps
from numpy.typing import NDArray

from pypolymlp.core.units import EVtoGPa

from .lammps_structure import LammpsStructure


class LammpsCommand:
    """Class for controlling lammps commands."""

    def __init__(
        self,
        elements: list,
        pot: str = "polymlp.yaml",
        style: str = "polymlp",
        style_command: str = "pair_style",
        coeff_command: str = "pair_coeff",
        stress: bool = True,
        log: bool = False,
        args: Optional[dict] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        structure: Initial structure in LammpsStructure.
        pot: Potential file name.
        stress: Compute stress.
        log: Generate log file.
        screen: Generate standard output to the screen.
        empirical_style:
        empirical_coeff:
        """
        self._elements = elements
        self._n_atomtypes = len(self._elements)
        self._pot = pot
        self._verbose = verbose

        cmdargs = []
        if not verbose:
            cmdargs.extend(["-screen", "none"])
        if not log:
            cmdargs.extend(["-log", "none"])

        self._lmp = lammps(cmdargs=cmdargs)
        self._lmp.command("dimension 3")
        self._lmp.command("units metal")
        self._lmp.command("boundary p p p")
        self._lmp.command("box tilt large")

        # Box initialization.
        # This must be called before providing potential style.
        self._lmp.command("region reg1 prism 0 10000 0 10000 0 10000 0 0 0")
        self._lmp.command("create_box " + str(self._n_atomtypes) + " reg1")

        # Style and coeff initialization.
        self._style = style_command + " " + style
        self._coeff = coeff_command + " * * " + pot + " " + " ".join(elements)
        self._lmp.command(self._style)
        self._lmp.command(self._coeff)
        if self._verbose:
            print("Lammps style:", self._style, flush=True)
            print("Lammps coeff:", self._coeff, flush=True)

        # property initialization
        # Number of neighbor atoms is appropriate? 5000? 10000?
        self._lmp.command("variable energy equal pe")
        self._lmp.command("variable volume equal vol")
        self._lmp.command("neigh_modify every 1 delay 0")
        self._lmp.command("neigh_modify one 2000")
        self._initialize_command_stress()

    def _initialize_command_stress(self):
        """Provide required commands for computing stress."""
        self._lmp.command("thermo 1")
        self._lmp.command(
            "thermo_style custom step temp pe press "
            + "pxx pyy pzz pxy pxz pyz lx ly lz vol"
        )
        self._lmp.command("thermo_modify norm no")

        self._lmp.command("variable pxx0 equal pxx")
        self._lmp.command("variable pyy0 equal pyy")
        self._lmp.command("variable pzz0 equal pzz")
        self._lmp.command("variable pyz0 equal pyz")
        self._lmp.command("variable pxz0 equal pxz")
        self._lmp.command("variable pxy0 equal pxy")

        self._lmp.command("variable lx0 equal lx")
        self._lmp.command("variable ly0 equal ly")
        self._lmp.command("variable lz0 equal lz")

    def eval(self, lmp_st: LammpsStructure):
        """Compute enthalpy, forces, and stress.

        Return
        ------
        energy: Energy. Unit: eV/cell.
        forces: Forces. shape=(3, N), Unit: eV/angstrom.
        stress: Stress tensor.
                shape=(6) in the order of xx, yy, zz, xy, yz, zx. Unit: eV/cell.
        """
        self.set_structure(lmp_st)
        self._lmp.command("run 0")
        return (self.energy, self.forces, self.stress)

    def set_structure(self, lmp_st: LammpsStructure):
        """Set structure."""
        lmp_st.recast_types(self._elements)
        for t in lmp_st.types:
            self._lmp.command("create_atoms " + str(t + 1) + " single 0 0 0")
        self.change_structure(lmp_st.axis, lmp_st.positions_cartesian)

    def change_structure(self, axis: NDArray, positions_c: NDArray):
        """Change structure using 6-dimensional axis vector and cartesian positions.

        Caution: The order in types must be considered.
        """
        self.change_box(axis)
        for i, pos in enumerate(positions_c.T):
            pos_str = " x " + str(pos[0]) + " y " + str(pos[1]) + " z " + str(pos[2])
            self._lmp.command("set atom " + str(i + 1) + pos_str)

    def change_box(self, axis: list | tuple | NDArray):
        """Change box using 6-dimensional axis vector."""
        lx, ly, lz, xy, yz, xz = axis
        self._lmp.command(
            "change_box all x final 0.0 "
            + str(lx)
            + " y final 0.0 "
            + str(ly)
            + " z final 0.0 "
            + str(lz)
            + " xy final "
            + str(xy)
            + " xz final "
            + str(xz)
            + " yz final "
            + str(yz)
        )

    @property
    def elements(self):
        """Return unique elements."""
        return self._elements

    @property
    def energy(self):
        """Return energy."""
        energy = self._lmp.extract_variable("energy", None, 0)
        return energy

    @property
    def forces(self):
        """Return forces. shape = (3, N)."""
        n = self._lmp.get_natoms()
        f = self._lmp.extract_atom("f", 3)
        forces = np.array([[f[i][0], f[i][1], f[i][2]] for i in range(n)]).T
        return forces

    @property
    def stress(self):
        """Get stress and lattice parameters."""
        pxx = self._lmp.extract_variable("pxx0", None, 0)
        pyy = self._lmp.extract_variable("pyy0", None, 0)
        pzz = self._lmp.extract_variable("pzz0", None, 0)
        pxy = self._lmp.extract_variable("pxy0", None, 0)
        pyz = self._lmp.extract_variable("pyz0", None, 0)
        pxz = self._lmp.extract_variable("pxz0", None, 0)
        unit = self.volume / EVtoGPa / 10000
        return np.array([pxx, pyy, pzz, pxy, pyz, pxz]) * unit

    @property
    def volume(self):
        """Return volume."""
        volume = self._lmp.extract_variable("volume", None, 0)
        return volume

    @property
    def box(self):
        """Return box size."""
        lx = self._lmp.extract_variable("lx0", None, 0)
        ly = self._lmp.extract_variable("ly0", None, 0)
        lz = self._lmp.extract_variable("lz0", None, 0)
        return np.array([lx, ly, lz])

    def command(self, string: str):
        """Provide a lammps command from string."""
        self._lmp.command(string)


# def extract_structure_from_lammps_obj(lmp):
#     """Extract lammps structure from lammps object."""
#
#     boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
#     xhi, yhi, zhi = np.array(boxhi) - np.array(boxlo)
#
#     n = lmp.get_natoms()
#     n_atomtypes = lmp.extract_global("ntypes", 0)
#
#     types = lmp.extract_atom("type", 0)
#     types1 = [types[i] - 1 for i in range(n)]
#
#     masses_lmp = lmp.extract_atom("mass", 2)
#     masses = dict()
#     for i in range(n_atomtypes):
#         masses[i] = masses_lmp[i + 1]
#
#     cds = lmp.extract_atom("x", 3)
#     positions_c = [[cds[i][0], cds[i][1], cds[i][2]] for i in range(n)]
#     positions_c = np.array([np.array(pos) - np.array(boxlo) for pos in positions_c]).T
#
#     axis = np.array([xhi, yhi, zhi, xy, yz, xz])
#     lmp_st = LammpsStructure(
#         axis=axis,
#         types=types1,
#         positions_cartesian=positions_c,
#         masses=masses,
#     )
#     return lmp_st
#

# def parse_lammps_structure(filename):
#     """Parse lammps structure file."""
#     lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
#     lmp.command("read_data " + filename)
#     lmp_st = extract_structure_from_lammps_obj(lmp)
#     return lmp_st
