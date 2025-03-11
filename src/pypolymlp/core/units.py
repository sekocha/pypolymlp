"""Unit conversion."""

# Avogadro constant (1/mol)
Avogadro = 6.02214076e23

# Planck constant  (6.62607015e−34 J*s)
Planck = 6.62607015e-34

# Boltzmann constant (1.380649e-23 J/K)
Boltzmann = 1.380649e-23

# Electron volt in J (1 eV = 1.602176634e-19 J)
EVtoJ = 1.602176634e-19

# Convert eV to kJ/mol
EVtoKJmol = 96.48533212331002

# Convert eV to kbar and eV to GPa
EVtoKbar = 1602.1766208
EVtoGPa = 160.21766208


def kjmol_to_ev(e):
    return e / 96.48533212331002


def ev_to_kjmol(e):
    return e * 96.48533212331002
