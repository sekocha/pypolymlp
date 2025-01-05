"""Unit conversion."""

# Avogadro constant
avogadro = 6.02214076e23

# Electron volt in J
EVtoJ = 1.602176634e-19

# Convert eV to kJ/mol
EVtoKJmol = 96.48533212331002


def kjmol_to_ev(e):
    return e / 96.48533212331002


def ev_to_kjmol(e):
    return e * 96.48533212331002
