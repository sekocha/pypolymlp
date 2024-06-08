#!/usr/bin/env python
import sys

import numpy as np


class Log:

    def __init__(self, name):

        f = open(name)
        self.lines = f.readlines()
        f.close()

    def get_energy(self, single=False, enthalpy=False):
        init_energy, final_energy = 0.0, 0.0
        if single:
            for i, line in enumerate(self.lines):
                if "Step Temp E_pair E_mol TotEng Press" in line:
                    arr = self.lines[i + 1].split()
                    init_energy, final_energy = float(arr[4]), float(arr[4])
        elif enthalpy:
            for i, line in enumerate(self.lines):
                if "Energy initial, next-to-last, final" in line:
                    arr = self.lines[i + 1].split()
                    init_energy, final_energy = float(arr[0]), float(arr[2])
        else:
            for i, line in enumerate(self.lines):
                if "Step Temp E_pair E_mol TotEng Press" in line:
                    arr = self.lines[i + 1].split()
                    init_energy = float(arr[4])
                    arr = self.lines[i + 2].split()
                    final_energy = float(arr[4])
                    self.volume = float(arr[6])

        return init_energy, final_energy

    def get_press(self, single=False):
        init_press, final_press = 0.0, 0.0
        if single:
            for i, line in enumerate(self.lines):
                if "Step Temp E_pair E_mol TotEng Press" in line:
                    arr = self.lines[i + 1].split()
                    init_press, final_press = float(arr[5]), float(arr[5])
        else:
            for i, line in enumerate(self.lines):
                if "Step Temp E_pair E_mol TotEng Press" in line:
                    arr = self.lines[i + 1].split()
                    init_press = float(arr[5])
                    arr = self.lines[i + 2].split()
                    final_press = float(arr[5])

        return init_press, final_press

    def get_iterations(self):
        iteration = 0
        for i, line in enumerate(self.lines):
            if "Iterations, force evaluations" in line:
                iteration = int(line.split()[4])

        return iteration

    def get_time(self):
        for i, line in enumerate(self.lines):
            if "Pair    |" in line:
                time = float(line.split()[2])
        return time

    def get_volume(self):
        return self.volume


class Dump:

    def __init__(self, name):

        f = open(name)
        self.lines = f.readlines()
        f.close()

        index = self.lines.index("ITEM: NUMBER OF ATOMS\n")
        natom = int(self.lines[index + 1])

        self.axis = np.zeros((3, 3))

        iaxis = ["ITEM: BOX BOUNDS" in line for line in self.lines].index(True)
        data = np.array(
            [
                [float(a) for a in line.split()]
                for line in self.lines[iaxis + 1 : iaxis + 4]
            ]
        )
        if data.shape[1] == 3:
            xy, xz, yz = data[0][2], data[1][2], data[2][2]
        else:
            xy, xz, yz = 0, 0, 0

        xlo = data[0][0] - min([0.0, xy, xz, xy + xz])
        xhi = data[0][1] - max([0.0, xy, xz, xy + xz])
        ylo = data[1][0] - min([0.0, yz])
        yhi = data[1][1] - max([0.0, yz])
        zlo, zhi = data[2][0], data[2][1]

        self.axis[0][0] = xhi - xlo
        self.axis[0][1], self.axis[1][1] = xy, yhi - ylo
        self.axis[0][2], self.axis[1][2], self.axis[2][2] = (
            xz,
            yz,
            zhi - zlo,
        )

        iatoms = ["ITEM: ATOMS" in line for line in self.lines].index(True)
        #        n_types = len(set([int(a) for line in self.lines\
        #            [iatoms+1:iatoms+natom+1] for a in line.split()[1]]))
        n_types = max(
            [
                int(a)
                for line in self.lines[iatoms + 1 : iatoms + natom + 1]
                for a in line.split()[1]
            ]
        )

        positions, self.n_atoms = [[] for i in range(n_types)], [
            0 for i in range(n_types)
        ]
        for line in self.lines[iatoms + 1 : iatoms + natom + 1]:
            split = line.split()
            t, pos = int(split[1]) - 1, [float(a) for a in split[2:5]]
            positions[t].append(pos)
            self.n_atoms[t] += 1
        data = [p for post in positions for p in post]

        self.types = [i for i, p in enumerate(positions) for j in range(len(p))]

        inv = np.linalg.inv(self.axis)
        self.positions = np.dot(inv, np.array(data).T)

        self.st = dict()
        self.st["axis"] = self.axis
        self.st["positions"] = self.positions
        self.st["n_atoms"] = self.n_atoms
        self.st["types"] = self.types
        self.st["elements"] = None
        self.st["volume"] = None

    def get_axis(self):
        return self.axis

    def get_positions(self):
        return self.positions

    def get_n_atoms(self):
        return self.n_atoms

    def get_structure(self):
        return self.st


class DumpForce:

    def __init__(self, name):

        f = open(name)
        self.lines = f.readlines()
        f.close()

        index = self.lines.index("ITEM: NUMBER OF ATOMS\n")
        natom = int(self.lines[index + 1])

        iatoms = ["ITEM: ATOMS" in line for line in self.lines].index(True)

        self.force = []
        for line in self.lines[iatoms + 1 : iatoms + natom + 1]:
            split = line.split()
            self.force.append([float(a) for a in split[2:5]])
        self.force = np.array(self.force)

    def get_force(self):
        return self.force


if __name__ == "__main__":
    log = Log(sys.argv[1])
    init_energy, final_energy = log.get_energy()
    volume = log.get_volume()
    print(volume, final_energy)
