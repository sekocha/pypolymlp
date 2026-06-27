"""Classes and functions used for setting structure in lammps format."""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure


def convert_structure_to_lammps_format(structure: PolymlpStructure):
    """Convert structure in PolymlpStructure to structure in LammpsStructure."""
    axis = structure.axis
    a, b, c = np.linalg.norm(axis, axis=0)
    calpha = (axis[:, 1] @ axis[:, 2]) / (b * c)
    cbeta = (axis[:, 2] @ axis[:, 0]) / (c * a)
    cgamma = (axis[:, 0] @ axis[:, 1]) / (a * b)

    lx = a
    xy = b * cgamma
    xz = c * cbeta
    ly = np.sqrt(b * b - xy * xy)
    yz = (b * c * calpha - xy * xz) / ly
    lz = np.sqrt(c * c - xz * xz - yz * yz)

    lmp_axis = np.array([lx, ly, lz, xy, yz, xz])
    lmp_axis_full = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])
    rotation = lmp_axis_full @ np.linalg.inv(axis)

    lmp_st = LammpsStructure(
        axis=lmp_axis,
        types=structure.types,
        elements=structure.elements,
        positions=structure.positions,
        rotation=rotation,
    )
    return lmp_st


@dataclass
class LammpsStructure:
    """Dataclass of structure in lammps_format.

    Parameters
    ----------
    axis: Axis vector [lx, ly, lz, xy, yz, zx], shape=(6).
    types: Atomic type integers, (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
    positions: Scaled positions, shape=(3, n_atom).
    positions_cartesian: Cartesian positions, shape=(3, n_atom).

    (optional)
    elements: Element list, (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
    rotation: Rotation matrix from original structure. lmp_axis = rot @ axis.
    """

    axis: np.ndarray
    types: Union[np.ndarray, list]

    positions: Optional[np.ndarray] = None
    positions_cartesian: Optional[np.ndarray] = None

    elements: Optional[Union[np.ndarray, list]] = None
    n_atomtypes: Optional[int] = None

    axis_matrix: Optional[np.ndarray] = None
    axis_matrix_inverse: Optional[np.ndarray] = None

    rotation: Optional[np.ndarray] = None
    rotation_inverse: Optional[np.ndarray] = None

    verbose: bool = False

    def __post_init__(self):
        """Init method."""
        self.n_atomtypes = len(np.unique(self.types))

        self._axis_check()
        self._position_check()
        self._set_rotation_matrices()

    def _axis_check(self):
        if len(self.axis) != 6:
            raise ValueError("Number of axis elements must be 6.")

        a = self.axis
        self.axis_matrix = np.array([[a[0], a[3], a[5]], [0, a[1], a[4]], [0, 0, a[2]]])
        try:
            self.axis_matrix_inverse = np.linalg.inv(self.axis_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("LammpsStructure: Invalid axis matrix.")

    def _position_check(self):
        if self.positions is None and self.positions_cartesian is None:
            raise ValueError("positions or positions_cartesian required.")

        if self.positions is not None:
            self.positions_cartesian = self.axis_matrix @ np.array(self.positions)
        else:
            self.positions = self.axis_matrix_inverse @ self.positions_cartesian

    def _set_rotation_matrices(self):
        if self.rotation is None:
            self.rotation = np.eye(3)
            self.rotation_inverse = np.eye(3)
        else:
            self.rotation_inverse = np.linalg.inv(self.rotation)

    def to_initial_basis(self, lmp_cartesian: np.ndarray):
        """Return fractional coordinates in initial basis."""
        if self.rotation_inverse is None:
            raise ValueError("No definition of inverse rotation.")
        return self.rotation_inverse @ lmp_cartesian

    @property
    def lx(self):
        """Return lx."""
        return self.axis_matrix[0, 0]

    @property
    def ly(self):
        """Return ly."""
        return self.axis_matrix[1, 1]

    @property
    def lz(self):
        """Return lz."""
        return self.axis_matrix[2, 2]

    @property
    def xy(self):
        """Return xy."""
        return self.axis_matrix[0, 1]

    @property
    def yz(self):
        """Return yz."""
        return self.axis_matrix[1, 2]

    @property
    def xz(self):
        """Return xz."""
        return self.axis_matrix[0, 2]

    @property
    def volume(self) -> float:
        """Return volume."""
        return np.linalg.det(self.axis_matrix)


#    def _set_masses(self):
#        """Set masses."""
#        if self.elements is not None:
#            table = mass_table()
#            self.masses = dict()
#            for t, ele in zip(self.types, self.elements):
#                self.masses[t] = table[ele]

#    def _sort_atoms(self):
#        """Sort atoms in structure."""
#        sorted_types = sorted([(t, i) for i, t in enumerate(self.types)])
#        self.types = [t for t, i in sorted_types]
#        order_atoms = np.array([i for t, i in sorted_types])
#
#        if self.elements is not None:
#            self.elements = [self.elements[i] for i in order_atoms]
#
#        if self.positions is not None:
#            self.positions = np.array(self.positions)[:, order_atoms]
#            self.positions_cartesian = self.axis_matrix @ self.positions
#        elif self.positions_cartesian is not None:
#            self.positions_cartesian = np.array(self.positions_cartesian)[
#                :, order_atoms
#            ]
#            self.positions = self.axis_matrix_inverse @ self.positions_cartesian
#        return order_atoms

#    def set_elements(self, uniq_elements: Union[np.ndarray, list]):
#        """Set elements."""
#        if len(uniq_elements) != self.n_atomtypes:
#            raise ValueError(
#                "LammpsStructure: len(elements) != n_atomtypes in set_masses."
#            )
#        self.elements = [uniq_elements[t] for t in self.types]
#        self._set_masses()
#
#    def recast_types(self, uniq_elements: Union[np.ndarray, list]):
#        """Recast types and related variables using unique elements."""
#        if self.elements is None:
#            raise AttributeError("Elements are not defined.")
#
#        self.n_atomtypes = len(uniq_elements)
#        map1 = dict()
#        for i, ele in enumerate(uniq_elements):
#            map1[ele] = i
#        self.types = [map1[ele] for ele in self.elements]
#        self._set_masses()

#    @property
#    def structure(self) -> PolymlpStructure:
#        """Return structure in PolymlpStructure format."""
#        if self.elements is None:
#            raise AttributeError("elements attribute is required.")
#        if self.positions is None:
#            raise AttributeError("positions attribute is required.")
#
#        self._order_atoms_polymlp = self._sort_atoms()
#        n_atoms = collections.Counter(self.types)
#        n_atoms = [n_atoms[t] for t in range(self.n_atomtypes)]
#        st = PolymlpStructure(
#            axis=self.axis_matrix,
#            positions=self.positions,
#            n_atoms=n_atoms,
#            elements=self.elements,
#            types=self.types,
#        )
#        return st

#    @property
#    def lattice_parameters(self) -> tuple[float]:
#        """Return lattice parameters (a, b, c, alpha, beta, gamma)."""
#        metric = self.axis_matrix.T @ self.axis_matrix
#        a, b, c = sqrt(metric[0, 0]), sqrt(metric[1, 1]), sqrt(metric[2, 2])
#        alpha = degrees(acos(metric[1, 2] / (b * c)))
#        beta = degrees(acos(metric[0, 2] / (a * c)))
#        gamma = degrees(acos(metric[0, 1] / (a * b)))
#        return (a, b, c, alpha, beta, gamma)

#    def change_isotropic(self, eps: float = 1.0):
#        """Change axis isotropically."""
#        self.axis_matrix = self.axis_matrix * pow(eps, 0.333333333333)
#        self.positions_cartesian = self.axis_matrix @ self.positions
#        self.axis = [
#            self.axis_matrix[0, 0],
#            self.axis_matrix[1, 1],
#            self.axis_matrix[2, 2],
#            self.axis_matrix[0, 1],
#            self.axis_matrix[1, 2],
#            self.axis_matrix[0, 2],
#        ]

#    def print_poscar(
#        self,
#        filename: str = "poscar-lammps-python",
#        header: str = "Generated by lammps-python",
#        refine: bool = False,
#        symprec: float = 1e-3,
#    ):
#        """Generate POSCAR."""
#        st = self.structure
#        if refine:
#            st = SymCell(st=st, symprec=symprec).refine_cell()
#        write_poscar_file(st, filename=filename, header=header)
#        if self.verbose:
#            print("Atomic order in", filename, ":", flush=True)
#            print(self._order_atoms_polymlp)
#
#    def print_lammps_structure(
#        self,
#        filename: str = "structure",
#        charge: Optional[list] = None,
#    ):
#        """Generate Lammps structure file."""
#        f = open(filename, "w")
#
#        print("# Generated lammps structure file by LammpsFormat\n", file=f)
#        print(self.positions.shape[1], "atoms", file=f)
#        print(self.n_atomtypes, "atom types\n", file=f)
#        print("0.0", "{0:.15f}".format(self.axis_matrix[0, 0]), "xlo xhi", file=f)
#        print("0.0", "{0:.15f}".format(self.axis_matrix[1, 1]), "ylo yhi", file=f)
#        print("0.0", "{0:.15f}".format(self.axis_matrix[2, 2]), "zlo zhi\n", file=f)
#        print(
#            "{0:.15f}".format(self.axis_matrix[0, 1]),
#            "{0:.15f}".format(self.axis_matrix[0, 2]),
#            "{0:.15f}".format(self.axis_matrix[1, 2]),
#            "xy xz yz\n",
#            file=f,
#        )
#
#        print("Atoms\n", file=f)
#        if charge is None:
#            for i, pos in enumerate(self.positions_cartesian.T):
#                print(
#                    i + 1,
#                    self.types[i] + 1,
#                    "{0:.15f}".format(pos[0]),
#                    "{0:.15f}".format(pos[1]),
#                    "{0:.15f}".format(pos[2]),
#                    file=f,
#                )
#        else:
#            for i, pos in enumerate(self.positions_cartesian.T):
#                print(
#                    i + 1,
#                    self.types[i] + 1,
#                    charge[self.types[i]],
#                    "{0:.15f}".format(pos[0]),
#                    "{0:.15f}".format(pos[1]),
#                    "{0:.15f}".format(pos[2]),
#                    file=f,
#                )
#
#        if self.masses is not None:
#            print("", file=f)
#            print("Masses\n", file=f)
#            for i, mass in self.masses.items():
#                print(i + 1, mass, file=f)
#
#        f.close()
#

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


# def generate_lammps_structure_from_file(
#     lammps: str = None,
#     poscar: str = None,
#     elements: Union[list, np.ndarray] = None,
# ):
#     """Construct LammpsStructure from a file."""
#     if lammps is not None:
#         lmp_st = parse_lammps_structure(lammps)
#         if elements is not None:
#             lmp_st.set_elements(elements)
#     elif poscar is not None:
#         st = Poscar(poscar).structure
#         lmp_st = convert_structure_to_lammps_format(st)
#
#     return lmp_st
