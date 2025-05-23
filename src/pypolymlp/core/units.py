"""Unit conversion."""

# Avogadro constant (1/mol)
Avogadro = 6.02214076e23

# Planck constant  (6.62607015e−34 J*s)
Planck = 6.62607015e-34

# Boltzmann constant (1.380649e-23 J/K, 8.617389435726849e-05 eV/K)
Kb = 1.380649e-23
KbEV = 8.617389435726849e-05

# Electron volt in J (1 eV = 1.602176634e-19 J)
EVtoJ = 1.602176634e-19

# Convert eV to kJ/mol
EVtoKJmol = 96.48533212331002
EVtoJmol = 96485.33212331002

# Convert eV to kbar and eV to GPa
EVtoKbar = 1602.1766208
EVtoGPa = 160.21766208

# Convert atomic mass to kg
MasstoKG = 1e-3 / Avogadro

# Convert m/s to angstrom/fs
M_StoAng_Fs = 1e-5

# Convert Hartree to eV
HartreetoEV = 27.211386245981

# Convert Bohr to angstrom
BohrtoAng = 0.529177210544


def kjmol_to_ev(e):
    return e / 96.48533212331002


def ev_to_kjmol(e):
    return e * 96.48533212331002
