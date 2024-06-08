#!/usr/bin/env python
import numpy as np


class ACF:
    def __init__(self, filename="ACF.dat"):
        f = open(filename)
        lines2 = f.readlines()
        f.close()
        self.charge = []
        for line in lines2:
            split = line.split()
            if split[0].isdecimal():
                self.charge.append(float(split[4]))
        self.charge = np.asarray(self.charge)

    def get_charge(self):
        return self.charge

    def print_charge(self):
        np.savetxt("bader_charge", charge)


class BCF:
    def __init__(self, filename="BCF.dat"):
        f = open(filename)
        lines2 = f.readlines()
        f.close()

        n_atom = max([int(line.split()[5]) for line in lines2[2:-1]])

        self.charge = np.zeros(n_atom)
        for line in lines2[2:-1]:
            split = line.split()
            c = float(split[4])
            atom_idx = int(split[5]) - 1
            self.charge[atom_idx] += c

    def get_charge(self):
        return self.charge

    def print_charge(self):
        np.savetxt("bader_charge", charge)


if __name__ == "__main__":
    # acf = ACF()
    # charge = acf.get_charge()
    # print(charge)
    bcf = BCF()
    charge = bcf.get_charge()
    print(charge)
