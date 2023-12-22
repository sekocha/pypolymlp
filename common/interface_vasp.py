#!/usr/bin/env python
import numpy as np
import re
import xml.etree.ElementTree as ET

def parse_vaspruns(vaspruns, element_order=None):

    kbar_to_eV = 1 / 1602.1766208
    dft_dict = defaultdict(list)
    for vasp in vaspruns:
        v = Vasprun(vasp)
        property_dict = v.get_properties()
        structure_dict = v.get_structure()

        if element_order is not None:
            structure_dict, property_dict['force'] \
                    = permute_atoms(structure_dict,
                                    property_dict['force'],
                                    element_order)

        dft_dict['energy'].append(property_dict['energy'])
        force_ravel = np.ravel(property_dict['force'], order='F')
        dft_dict['force'].extend(force_ravel)

        sigma = property_dict['stress'] * structure_dict['volume'] * kbar_to_eV
        s = [sigma[0][0], sigma[1][1], sigma[2][2],
             sigma[0][1], sigma[1][2], sigma[2][0]]
        dft_dict['stress'].extend(s)
        dft_dict['structures'].append(structure_dict)

    dft_dict['energy'] = np.array(dft_dict['energy'])
    dft_dict['force'] = np.array(dft_dict['force'])
    dft_dict['stress'] = np.array(dft_dict['stress'])

    elements_size = [len(st['elements']) for st in dft_dict['structures']]
    elements = dft_dict['structures'][np.argmax(elements_size)]['elements']
    dft_dict['elements'] = sorted(set(elements), key=elements.index)

    dft_dict['total_n_atoms'] = np.array([sum(st['n_atoms'])
                                         for st in dft_dict['structures']])
    dft_dict['filenames'] = vaspruns

    return dft_dict


class Vasprun:

    def __init__(self, name):
        self._root = ET.parse(name).getroot()

    def get_energy(self):
        e = self._root.find('calculation').find('energy')
        return float(e[1].text)

    def get_energy_smearing_delta(self):
        e = self._root.find('calculation').find('energy')
        return float(e[2].text)

    def get_forces(self):
        f = self._root.find(".//*[@name='forces']")
        return self.__varray_to_nparray(f).T

    def get_stress(self): # unit: kbar
        f = self._root.find(".//*[@name='stress']")
        return self.__varray_to_nparray(f)

    def get_properties(self):
        property_dict = dict()
        property_dict['energy'] = self.get_energy()
        property_dict['force'] = self.get_forces()
        property_dict['stress'] = self.get_stress()
        return property_dict

    def get_structure(self, valence=False, key=None):
        st = self._root.find(".//*[@name='finalpos']")
        st1  = st.find(".//*[@name='basis']")
        st2  = st.find(".//*[@name='positions']")
        st3  = st.find(".//*[@name='volume']")
        st4  = self._root.findall(".//*[@name='atomtypes']/set/rc")
        st5  = self._root.findall(".//*[@name='atoms']/set/rc")

        structure_dict = dict()
        structure_dict['axis'] = self.__varray_to_nparray(st1).T
        structure_dict['positions'] = self.__varray_to_nparray(st2).T
        structure_dict['volume'] = float(st3.text)

        tmp1 = self.__read_rc_set(st4)
        structure_dict['n_atoms'] = [int(x) for x in list(np.array(tmp1)[:,0])]

        tmp2 = self.__read_rc_set(st5)
        structure_dict['elements'] = list(np.array(tmp2)[:,0])
        structure_dict['elements'] = ['Zr' if e == 'r' else e 
                                     for e in structure_dict['elements']]
        structure_dict['types'] = [int(x)-1 for x in list(np.array(tmp2)[:,1])]

        if valence:
            valence_dict = dict()
            for d in tmp1:
                valence_dict[d[1]] = float(d[3])
            structure_dict['valence'] = [dict_valence[e] for e in self.elements]

        if key is not None:
            return structure_dict[key]
        return structure_dict

    def __varray_to_nparray(self, varray):
        nparray = [[float(x) for x in v1.text.split()] for v1 in varray]
        nparray = np.array(nparray)
        return nparray

    def __read_rc_set(self, obj):
        rc_set = []
        for rc in obj:
            c_set = [c.text.replace(" ", "") for c in rc.findall('c')]
            rc_set.append(c_set)
        return rc_set


class Poscar:

    def __init__(self, filename, selective_dynamics=False):

        self.structure = dict()

        f = open(filename, 'r')
        lines = f.readlines()
        f.close()

        self.structure['comment'] = lines[0]

        axis_const = float(lines[1].split()[0])
        axis1 = [float(x) for x in lines[2].split()[0:3]]
        axis2 = [float(x) for x in lines[3].split()[0:3]]
        axis3 = [float(x) for x in lines[4].split()[0:3]]
        self.structure['axis'] = np.c_[axis1, axis2, axis3] * axis_const

        if (len(re.findall(r'[a-z,A-Z]+', lines[5])) > 0):
            uniq_elements = lines[5].split()
            self.structure['n_atoms'] = [int(x) for x in lines[6].split()]
            n_line = 7 
        else:
            uniq_elements = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            self.structure['n_atoms'] = [int(x) for x in lines[5].split()]
            n_line = 6

        self.structure['elements'] = []
        self.structure['types'] = []
        for i, n in enumerate(self.structure['n_atoms']):
            for j in range(n):
                self.structure['types'].append(i)
                self.structure['elements'].append(uniq_elements[i])

        if selective_dynamics:
            sd = lines[begin_nline]
            n_line += 1

        coord_type = lines[n_line].split()[0]
        n_line += 1

        positions = []
        for i in range(sum(self.structure['n_atoms'])):
            pos = [float(x) for x in lines[n_line].split()[0:3]]
            positions.append(pos)
            n_line += 1
        self.structure['positions'] = np.array(positions).T

        self.structure['volume'] = np.linalg.det(self.structure['axis'])

    def get_structure(self):
        return self.structure


