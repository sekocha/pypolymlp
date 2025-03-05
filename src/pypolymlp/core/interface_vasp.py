"""Interfaces for vasp."""

import re
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpDataDFT, PolymlpStructure
from pypolymlp.core.interface_datasets import set_dataset_from_structures
from pypolymlp.core.units import EVtoKbar


def set_dataset_from_vaspruns(
    vaspruns: list[str],
    element_order: Optional[bool] = None,
) -> PolymlpDataDFT:
    """Return DFT dataset by loading vasprun.xml files."""
    structures, (energies, forces, stresses) = parse_properties_from_vaspruns(vaspruns)
    dft = set_dataset_from_structures(
        structures,
        energies,
        forces=forces,
        stresses=stresses,
        element_order=element_order,
    )
    return dft


def parse_properties_from_vaspruns(vaspruns: list[str]) -> tuple:
    """Parse vasprun.xml files and return structures and properties."""
    energies, forces, stresses, structures = [], [], [], []
    for vasp in vaspruns:
        md, root = check_vasprun_type(vasp)
        if md:
            v = VasprunMD(vasp)
            structures.extend(v.structures)
            energies.extend(v.energies)
            forces.extend(v.forces)
            for stress, st in zip(v.stresses, v.structures):
                sigma = stress * st.volume / EVtoKbar
                stresses.append(sigma)
        else:
            v = Vasprun(vasp)
            structures.append(v.structure)
            energies.append(v.energy)
            forces.append(v.forces)
            sigma = v.stress * v.structure.volume / EVtoKbar
            stresses.append(sigma)
    return structures, (np.array(energies), forces, np.array(stresses))


def parse_structures_from_vaspruns(vaspruns: list[str]) -> list[PolymlpStructure]:
    """Parse vasprun.xml files and return structures."""
    return [Vasprun(f).structure for f in vaspruns]


def parse_structures_from_poscars(poscars: list[str]):
    """Parse POSCAR files and return structures."""
    return [Poscar(f).structure for f in poscars]


def check_vasprun_type(name: str = None, root=None):
    """Check whether md type calculation is done in vasprun.xml."""
    if name is not None:
        root = ET.parse(name).getroot()

    tag = root.find(".//*[@name='IBRION']")
    try:
        tagint = int(tag.text)
        if tagint == 0:
            return True, root
        return False, root
    except:
        return False, root


class Vasprun:
    """Class for parsing vasprun.xml from single-point calculation."""

    def __init__(self, name: str, root=None):
        """Init method."""
        if root is None:
            self._root = ET.parse(name).getroot()
        else:
            self._root = root
        self._calc = self._root.find("calculation")

        self._energy = None
        self._forces = None
        self._stress = None
        self._structure = None
        self._name = name

    def get_energy_smearing_delta(self) -> float:
        """Parse vasprun and return smearing delta F."""
        e = self._calc.find("energy")
        return float(e[2].text)

    @property
    def energy(self) -> float:
        """Parse vasprun and return energy.

        Return
        ------
        energy: float.
        """
        if self._energy is not None:
            return self._energy
        e = self._calc.find("energy")
        self._energy = float(e[1].text)
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """Parse vasprun and return forces.

        Return
        ------
        forces: shape=(3, n_atom)
        """
        if self._forces is not None:
            return self._forces
        f = self._root.find(".//*[@name='forces']")
        self._forces = self._varray_to_nparray(f).T
        return self._forces

    @property
    def stress(self) -> np.ndarray:
        """Parse vasprun and return stress tensor in kbar.

        Return
        ------
        stress: shape=(3, 3) in kbar.
        """
        if self._stress is not None:
            return self._stress
        f = self._root.find(".//*[@name='stress']")
        self._stress = self._varray_to_nparray(f)
        return self._stress

    @property
    def properties(self) -> tuple:
        """Return properties."""
        return (self.energy, self.forces, self.stress)

    @property
    def structure(self) -> PolymlpStructure:
        """Parse vasprun and return structure."""
        if self._structure is not None:
            return self._structure

        st = self._root.find(".//*[@name='finalpos']")
        st1 = st.find(".//*[@name='basis']")
        st2 = st.find(".//*[@name='positions']")
        st3 = st.find(".//*[@name='volume']")
        st4 = self._root.findall(".//*[@name='atomtypes']/set/rc")
        st5 = self._root.findall(".//*[@name='atoms']/set/rc")

        axis = self._varray_to_nparray(st1).T
        positions = self._varray_to_nparray(st2).T
        volume = float(st3.text)

        tmp1 = self._read_rc_set(st4)
        n_atoms = [int(x) for x in list(np.array(tmp1)[:, 0])]

        tmp2 = self._read_rc_set(st5)
        elements = list(np.array(tmp2)[:, 0])
        elements = ["Zr" if e == "r" else e for e in elements]
        types = [int(x) - 1 for x in list(np.array(tmp2)[:, 1])]

        # if valence:
        #     valence_dict = dict()
        #     for d in tmp1:
        #         valence_dict[d[1]] = float(d[3])
        #     valence = [valence_dict[e] for e in self.elements]
        # else:
        #     valence = None

        self._structure = PolymlpStructure(
            axis,
            positions,
            n_atoms,
            elements,
            types,
            volume,
            name=self._name,
        )
        return self._structure

    def get_scstep(self) -> np.ndarray:
        """Return SC step."""
        scsteps = self._root.find("calculation").findall("scstep")
        e_history = []
        for sc in scsteps:
            e0 = sc.find("energy").find(".//*[@name='e_0_energy']")
            e_history.append(float(e0.text))
        return np.array(e_history)

    def _varray_to_nparray(self, varray):
        """Convert varray to numpy array."""
        nparray = [[float(x) for x in v1.text.split()] for v1 in varray]
        return np.array(nparray)

    def _read_rc_set(self, obj):
        """Read rc_set."""
        return [[c.text.replace(" ", "") for c in rc.findall("c")] for rc in obj]


class VasprunMD:
    """Class for parsing vasprun.xml from MD."""

    def __init__(self, name: str, root=None):
        """Init method."""
        if root is None:
            self._root = ET.parse(name).getroot()
        else:
            self._root = root
        self._calcs = self._root.findall("calculation")

        self._energies = None
        self._forces = None
        self._stresses = None
        self._structures = None
        self._name = name

    @property
    def energies(self):
        """Return energies.

        Return
        ------
        energies: shape = (n_str)
        """
        if self._energies is not None:
            return self._energies

        tag = "energy"
        self._energies = [float(cal.find(tag)[1].text) for cal in self._calcs]
        self._energies = np.array(self._energies)
        return self._energies

    @property
    def forces(self):
        """Return forces.

        Return
        ------
        forces: shape = (n_str, 3, n_atom).
        """
        if self._forces is not None:
            return self._forces

        self._forces = []
        tag = ".//*[@name='forces']"
        for cal in self._calcs:
            f = self._varray_to_nparray(cal.find(tag)).T
            self._forces.append(f)
        return self._forces

    @property
    def stresses(self):
        """Return stress tensors.

        Return
        ------
        stresses: shape = (n_str, 3, 3)
        """
        if self._stresses is not None:
            return self._stresses

        self._stresses = []
        tag = ".//*[@name='stress']"
        for cal in self._calcs:
            s = self._varray_to_nparray(cal.find(tag))
            self._stresses.append(s)
        self._stresses = np.array(self._stresses)
        return self._stresses

    @property
    def structures(self):
        """Return structures.

        Return
        ------
        structures: list[PolymlpStrucuture]
        """
        if self._structures is not None:
            return self._structures

        rc = self._root.findall(".//*[@name='atomtypes']/set/rc")
        rc_set = self._read_rc_set(rc)
        n_atoms = [int(x) for x in list(np.array(rc_set)[:, 0])]

        # if valence:
        #     valence_dict = dict()
        #     for d in rc_set:
        #         valence_dict[d[1]] = float(d[3])
        #     valence = [valence_dict[e] for e in self.elements]
        # else:
        #     valence = None

        rc = self._root.findall(".//*[@name='atoms']/set/rc")
        rc_set = self._read_rc_set(rc)
        elements = list(np.array(rc_set)[:, 0])
        elements = ["Zr" if e == "r" else e for e in elements]
        types = [int(x) - 1 for x in list(np.array(rc_set)[:, 1])]

        self._structures = []
        for cal in self._calcs:
            st1 = cal.find(".//*[@name='basis']")
            st2 = cal.find(".//*[@name='positions']")
            st3 = cal.find(".//*[@name='volume']")
            axis = self._varray_to_nparray(st1).T
            positions = self._varray_to_nparray(st2).T
            volume = float(st3.text)

            st = PolymlpStructure(
                axis,
                positions,
                n_atoms,
                elements,
                types,
                volume,
                name=self._name,
            )
            self._structures.append(st)

        return self._structures

    def _varray_to_nparray(self, varray):
        """Convert varray to numpy array."""
        nparray = [[float(x) for x in v1.text.split()] for v1 in varray]
        return np.array(nparray)

    def _read_rc_set(self, obj):
        """Read rc_set."""
        return [[c.text.replace(" ", "") for c in rc.findall("c")] for rc in obj]


class Poscar:
    """Class for parsing POSCAR."""

    def __init__(self, filename: str, selective_dynamics: bool = False):
        """Init method."""
        self._parse(filename, selective_dynamics=selective_dynamics)

    def _parse(self, filename: str, selective_dynamics: bool = False):
        """Parse POSCAR file."""
        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        comment = lines[0].replace("\n", "")
        axis_const = float(lines[1].split()[0])
        axis1 = [float(x) for x in lines[2].split()[0:3]]
        axis2 = [float(x) for x in lines[3].split()[0:3]]
        axis3 = [float(x) for x in lines[4].split()[0:3]]
        axis = np.c_[axis1, axis2, axis3] * axis_const

        if len(re.findall(r"[a-z,A-Z]+", lines[5])) > 0:
            uniq_elements = lines[5].split()
            n_atoms = [int(x) for x in lines[6].split()]
            n_line = 7
        else:
            uniq_elements = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            n_atoms = [int(x) for x in lines[5].split()]
            n_line = 6

        elements, types = [], []
        for i, n in enumerate(n_atoms):
            for j in range(n):
                types.append(i)
                elements.append(uniq_elements[i])

        if selective_dynamics:
            # sd = lines[begin_nline]
            n_line += 1

        # coord_type = lines[n_line].split()[0]
        n_line += 1

        positions = []
        for i in range(sum(n_atoms)):
            pos = [float(x) for x in lines[n_line].split()[0:3]]
            positions.append(pos)
            n_line += 1
        positions = np.array(positions).T
        volume = np.linalg.det(axis)

        self._structure = PolymlpStructure(
            axis,
            positions,
            n_atoms,
            elements,
            types,
            volume,
            comment=comment,
            name=filename,
        )

    @property
    def structure(self) -> PolymlpStructure:
        """Return structure."""
        return self._structure


class Outcar:
    def __init__(self, outcar_name):
        self.outcar_name = outcar_name

    def parse_e(self):
        return float(self.grep("energy  without entropy", self.outcar_name)[6])

    def parse_ewald(self):
        return float(self.grep("electrostatic energy", self.outcar_name)[3])

    def parse_ewald2(self):
        self_e = float(self.grep("energy (self)", self.outcar_name)[3])
        self_r = float(self.grep("energy (reciprocal space)", self.outcar_name)[4])
        return self_e + self_r

    def grep(self, text, file_name):
        f = open(file_name)
        lines = f.readlines()
        f.close()
        for line in lines:
            if line.find(text) >= 0:
                return line[:-1].split()


def read_doscar(name):
    """Parse DOSCAR file."""
    f = open(name)
    lines = f.readlines()
    f.close()

    e_fermi = float(lines[5].split()[3])
    dos = []
    for l1 in lines[6:]:
        str1 = l1.split()[:2]
        vals = [float(str1[0]) - e_fermi, float(str1[1])]
        dos.append(vals)
    return np.array(dos)


def parse_energy_volume(vaspruns):
    """Parse energy-volume data from vaspruns."""
    ev_data = []
    for vasprun_file in vaspruns:
        vasp = Vasprun(vasprun_file)
        energy = vasp.energy
        vol = vasp.structure.volume
        ev_data.append([vol, energy])
    return np.array(ev_data)


# class Chg:
#
#     def __init__(self, fname="CHG"):
#         p = Poscar(fname)
#         self.axis, self.positions, n_atoms, elements, types = p.get_structure()
#         st = Structure(self.axis, self.positions, n_atoms, elements, types)
#         self.vol = st.calc_volume()
#
#         f = open(fname)
#         lines2 = f.readlines()
#         f.close()
#
#         start = sum(n_atoms) + 9
#         self.grid = [int(i) for i in lines2[start].split()]
#         self.ngrid = np.prod(self.grid)
#
#         chg = [float(s) for line in lines2[start + 1 :] for s in line.split()]
#         self.chg = np.array(chg) / self.ngrid
#         self.chgd = np.array(chg) / self.vol
#
#         grid_fracs = np.array(
#             [
#                 np.array([x[2], x[1], x[0]])
#                 for x in itertools.product(
#                     range(self.grid[2]), range(self.grid[1]), range(self.grid[0])
#                 )
#             ]
#         ).T
#
#         self.grid_fracs = [
#             grid_fracs[0, :] / self.grid[0],
#             grid_fracs[1, :] / self.grid[1],
#             grid_fracs[2, :] / self.grid[2],
#         ]
#
#     def get_grid(self):
#         return self.grid
#
#     def get_grid_coordinates(self):
#         self.grid_coordinates = np.dot(self.axis, self.grid_fracs)
#         return self.grid_coordinates
#
#     def get_grid_coordinates_atomcenter(self, atom):
#         pos1 = self.positions[:, atom]
#         frac_new = self.grid_fracs - np.tile(pos1, (self.grid_fracs.shape[1], 1)).T
#         frac_new[np.where(frac_new > 0.5)] -= 1.0
#         frac_new[np.where(frac_new < -0.5)] += 1.0
#         return np.dot(self.axis, frac_new)
#
#     def get_ngrid(self):
#         return self.ngrid
#
#     def get_chg(self):
#         return self.chg
#
#     def get_chg_density(self):
#         return self.chgd
#
#     def get_volume(self):
#         return self.vol
