"""Unit conversion."""

avogadro = 6.02214076e23
EVtoJ = 1.602176634e-19

EVtoKJmol = 96.48533212331002


def kjmol_to_ev(e):
    return e / 96.48533212331002


def ev_to_kjmol(e):
    return e * 96.48533212331002
