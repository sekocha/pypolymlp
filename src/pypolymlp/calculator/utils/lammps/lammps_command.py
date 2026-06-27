"""Class for controlling python lammps object."""

from typing import Optional

import numpy as np
from lammps import lammps
from numpy.typing import NDArray

from .lammps_structure import LammpsStructure

# from lammps_python.core.lammps_utils import (
#     extract_structure_from_lammps_obj,
# )


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
        self._style = style_command + " " + style
        self._coeff = coeff_command + " * * " + pot + " " + " ".join(elements)
        self._verbose = verbose
        if self._verbose:
            print("Lammps style:", self._style, flush=True)
            print("Lammps coeff:", self._coeff, flush=True)

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

        # box and structure initialization
        self._lmp.command("region reg1 prism 0 10000 0 10000 0 10000 0 0 0")
        self._lmp.command("create_box " + str(self._n_atomtypes) + " reg1")

        self._lmp.command(self._style)
        self._lmp.command(self._coeff)

        # property initialization
        self._lmp.command("variable energy equal pe")
        self._lmp.command("variable volume equal vol")
        self._lmp.command("neigh_modify every 1 delay 0")
        # TODO: Number of neighbor atoms is appropriate? 5000? 10000?
        self._lmp.command("neigh_modify one 2000")

        self._initialize_command_stress()

    def eval(self, lmp_st: LammpsStructure):
        """Compute enthalpy and forces.

        Return
        ------
        energy: Energy.
        forces: Forces. shape=(N, 3).
        """
        # structure: LammpsStructure,
        # structure modification
        # lmp_st.recast_types(self._elements)
        for t in lmp_st.types:
            self._lmp.command("create_atoms " + str(t + 1) + " single 0 0 0")
        self.change_structure(lmp_st.axis, lmp_st.positions_cartesian)

        self._lmp.command("run 0")
        # enthalpy = self.get_enthalpy(pressure=pressure)
        # return enthalpy, self.forces

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

    def command(self, string: str):
        """Provide a lammps command from string."""
        self._lmp.command(string)

    def change_box(self, axis: list):
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

    def change_structure(self, axis: NDArray, positions_c: NDArray):
        """Change structure using 6-dimensional axis vector and cartesian positions.

        Caution: The order in types must be considered.
        """
        self.change_box(axis)
        for i, pos in enumerate(positions_c.T):
            pos_str = " x " + str(pos[0]) + " y " + str(pos[1]) + " z " + str(pos[2])
            self._lmp.command("set atom " + str(i + 1) + pos_str)

    def compute(self, pressure=0.0):
        """Compute enthalpy and forces.

        Return
        ------
        energy: Energy.
        forces: Forces. shape=(N, 3).
        """
        self._lmp.command("run 0")
        enthalpy = self.get_enthalpy(pressure=pressure)
        return enthalpy, self.forces

    #    @property
    #    def structure(self):
    #        """Return lammps structure."""
    #        lmp_st = extract_structure_from_lammps_obj(self._lmp)
    #        lmp_st.set_elements(self.elements)
    #        return lmp_st
    #
    #    @property
    #    def elements(self):
    #        """Return unique elements read from potential."""
    #        return self._elements

    @property
    def energy(self):
        """Return energy."""
        energy = self._lmp.extract_variable("energy", None, 0)
        return energy

    @property
    def forces(self):
        """Return forces. shape = (N, 3)."""
        n = self._lmp.get_natoms()
        f = self._lmp.extract_atom("f", 3)
        forces = np.array([[f[i][0], f[i][1], f[i][2]] for i in range(n)])
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
        lx = self._lmp.extract_variable("lx0", None, 0)
        ly = self._lmp.extract_variable("ly0", None, 0)
        lz = self._lmp.extract_variable("lz0", None, 0)
        return (pxx, pyy, pzz, pxy, pyz, pxz), (lx, ly, lz)

    @property
    def volume(self):
        """Return volume."""
        volume = self._lmp.extract_variable("volume", None, 0)
        return volume
