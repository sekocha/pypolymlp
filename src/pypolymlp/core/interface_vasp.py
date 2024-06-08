#!/usr/bin/env python
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from pypolymlp.core.utils import permute_atoms


def parse_vaspruns(vaspruns, element_order=None):

    kbar_to_eV = 1 / 1602.1766208
    dft_dict = defaultdict(list)
    for vasp in vaspruns:
        v = Vasprun(vasp)
        property_dict = v.get_properties()
        structure_dict = v.get_structure()

        if element_order is not None:
            structure_dict, property_dict["force"] = permute_atoms(
                structure_dict, property_dict["force"], element_order
            )

        dft_dict["energy"].append(property_dict["energy"])
        force_ravel = np.ravel(property_dict["force"], order="F")
        dft_dict["force"].extend(force_ravel)

        sigma = property_dict["stress"] * structure_dict["volume"] * kbar_to_eV
        s = [
            sigma[0][0],
            sigma[1][1],
            sigma[2][2],
            sigma[0][1],
            sigma[1][2],
            sigma[2][0],
        ]
        dft_dict["stress"].extend(s)
        dft_dict["structures"].append(structure_dict)
        dft_dict["volumes"].append(structure_dict["volume"])

    dft_dict["energy"] = np.array(dft_dict["energy"])
    dft_dict["force"] = np.array(dft_dict["force"])
    dft_dict["stress"] = np.array(dft_dict["stress"])
    dft_dict["volumes"] = np.array(dft_dict["volumes"])

    if element_order is None:
        elements_size = [len(st["elements"]) for st in dft_dict["structures"]]
        elements = dft_dict["structures"][np.argmax(elements_size)]["elements"]
        dft_dict["elements"] = sorted(set(elements), key=elements.index)
    else:
        dft_dict["elements"] = element_order

    dft_dict["total_n_atoms"] = np.array(
        [sum(st["n_atoms"]) for st in dft_dict["structures"]]
    )
    dft_dict["filenames"] = vaspruns

    return dft_dict


def parse_structures_from_vaspruns(vaspruns):
    return [Vasprun(f).get_structure() for f in vaspruns]


def parse_structures_from_poscars(poscars):
    return [Poscar(f).get_structure() for f in poscars]


class Vasprun:

    def __init__(self, name):
        self._root = ET.parse(name).getroot()

    def get_energy(self):
        e = self._root.find("calculation").find("energy")
        return float(e[1].text)

    def get_energy_smearing_delta(self):
        e = self._root.find("calculation").find("energy")
        return float(e[2].text)

    def get_forces(self):
        f = self._root.find(".//*[@name='forces']")
        return self.__varray_to_nparray(f).T

    def get_stress(self):  # unit: kbar
        f = self._root.find(".//*[@name='stress']")
        return self.__varray_to_nparray(f)

    def get_properties(self):
        property_dict = dict()
        property_dict["energy"] = self.get_energy()
        property_dict["force"] = self.get_forces()
        property_dict["stress"] = self.get_stress()
        return property_dict

    def get_structure(self, valence=False, key=None):
        st = self._root.find(".//*[@name='finalpos']")
        st1 = st.find(".//*[@name='basis']")
        st2 = st.find(".//*[@name='positions']")
        st3 = st.find(".//*[@name='volume']")
        st4 = self._root.findall(".//*[@name='atomtypes']/set/rc")
        st5 = self._root.findall(".//*[@name='atoms']/set/rc")

        structure_dict = dict()
        structure_dict["axis"] = self.__varray_to_nparray(st1).T
        structure_dict["positions"] = self.__varray_to_nparray(st2).T
        structure_dict["volume"] = float(st3.text)

        tmp1 = self.__read_rc_set(st4)
        structure_dict["n_atoms"] = [int(x) for x in list(np.array(tmp1)[:, 0])]

        tmp2 = self.__read_rc_set(st5)
        structure_dict["elements"] = list(np.array(tmp2)[:, 0])
        structure_dict["elements"] = [
            "Zr" if e == "r" else e for e in structure_dict["elements"]
        ]
        structure_dict["types"] = [int(x) - 1 for x in list(np.array(tmp2)[:, 1])]

        if valence:
            valence_dict = dict()
            for d in tmp1:
                valence_dict[d[1]] = float(d[3])
            structure_dict["valence"] = [valence_dict[e] for e in self.elements]

        if key is not None:
            return structure_dict[key]
        return structure_dict

    def get_scstep(self):
        scsteps = self._root.find("calculation").findall("scstep")
        e_history = []
        for sc in scsteps:
            e0 = sc.find("energy").find(".//*[@name='e_0_energy']")
            e_history.append(float(e0.text))
        return np.array(e_history)

    def __varray_to_nparray(self, varray):
        nparray = [[float(x) for x in v1.text.split()] for v1 in varray]
        nparray = np.array(nparray)
        return nparray

    def __read_rc_set(self, obj):
        rc_set = []
        for rc in obj:
            c_set = [c.text.replace(" ", "") for c in rc.findall("c")]
            rc_set.append(c_set)
        return rc_set


class Poscar:

    def __init__(self, filename, selective_dynamics=False):

        self.structure = dict()

        f = open(filename, "r")
        lines = f.readlines()
        f.close()

        self.structure["comment"] = lines[0].replace("\n", "")

        axis_const = float(lines[1].split()[0])
        axis1 = [float(x) for x in lines[2].split()[0:3]]
        axis2 = [float(x) for x in lines[3].split()[0:3]]
        axis3 = [float(x) for x in lines[4].split()[0:3]]
        self.structure["axis"] = np.c_[axis1, axis2, axis3] * axis_const

        if len(re.findall(r"[a-z,A-Z]+", lines[5])) > 0:
            uniq_elements = lines[5].split()
            self.structure["n_atoms"] = [int(x) for x in lines[6].split()]
            n_line = 7
        else:
            uniq_elements = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            self.structure["n_atoms"] = [int(x) for x in lines[5].split()]
            n_line = 6

        self.structure["elements"] = []
        self.structure["types"] = []
        for i, n in enumerate(self.structure["n_atoms"]):
            for j in range(n):
                self.structure["types"].append(i)
                self.structure["elements"].append(uniq_elements[i])

        if selective_dynamics:
            # sd = lines[begin_nline]
            n_line += 1

        # coord_type = lines[n_line].split()[0]
        n_line += 1

        positions = []
        for i in range(sum(self.structure["n_atoms"])):
            pos = [float(x) for x in lines[n_line].split()[0:3]]
            positions.append(pos)
            n_line += 1
        self.structure["positions"] = np.array(positions).T

        self.structure["volume"] = np.linalg.det(self.structure["axis"])

    def get_structure(self):
        return self.structure


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

    ev_data = []
    for vasprun_file in vaspruns:
        vasp = Vasprun(vasprun_file)
        energy = vasp.get_energy()
        vol = vasp.get_structure()["volume"]
        ev_data.append([vol, energy])
    return np.array(ev_data)


"""
class Chg:

    def __init__(self, fname="CHG"):
        p = Poscar(fname)
        self.axis, self.positions, n_atoms, elements, types = p.get_structure()
        st = Structure(self.axis, self.positions, n_atoms, elements, types)
        self.vol = st.calc_volume()

        f = open(fname)
        lines2 = f.readlines()
        f.close()

        start = sum(n_atoms) + 9
        self.grid = [int(i) for i in lines2[start].split()]
        self.ngrid = np.prod(self.grid)

        chg = [float(s) for line in lines2[start + 1 :] for s in line.split()]
        self.chg = np.array(chg) / self.ngrid
        self.chgd = np.array(chg) / self.vol

        grid_fracs = np.array(
            [
                np.array([x[2], x[1], x[0]])
                for x in itertools.product(
                    range(self.grid[2]), range(self.grid[1]), range(self.grid[0])
                )
            ]
        ).T

        self.grid_fracs = [
            grid_fracs[0, :] / self.grid[0],
            grid_fracs[1, :] / self.grid[1],
            grid_fracs[2, :] / self.grid[2],
        ]

    def get_grid(self):
        return self.grid

    def get_grid_coordinates(self):
        self.grid_coordinates = np.dot(self.axis, self.grid_fracs)
        return self.grid_coordinates

    def get_grid_coordinates_atomcenter(self, atom):
        pos1 = self.positions[:, atom]
        frac_new = self.grid_fracs - np.tile(pos1, (self.grid_fracs.shape[1], 1)).T
        frac_new[np.where(frac_new > 0.5)] -= 1.0
        frac_new[np.where(frac_new < -0.5)] += 1.0
        return np.dot(self.axis, frac_new)

    def get_ngrid(self):
        return self.ngrid

    def get_chg(self):
        return self.chg

    def get_chg_density(self):
        return self.chgd

    def get_volume(self):
        return self.vol
"""
