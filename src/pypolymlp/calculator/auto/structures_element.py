"""Functions for providing initial structures for systematic calculations."""

import numpy as np

from pypolymlp.calculator.auto.dataclass import Prototype
from pypolymlp.core.data_format import PolymlpStructure


def _get_structure_attrs(n_atom: int, element_strings: tuple):
    """Return structure attributes for elemental system."""
    n_atoms = [n_atom]
    types = np.zeros(n_atom, dtype=int)
    elements = [ele for n, ele in zip(n_atoms, element_strings) for _ in range(n)]
    return (n_atoms, types, elements)


def get_structure_list_element(element_strings: tuple):
    """Return structure list for elemental systems."""
    fcc = structure_fcc(element_strings, a=5.0)
    bcc = structure_bcc(element_strings, a=4.0)
    return [fcc, bcc]


def structure_fcc(element_strings: tuple, a: float = 5.0):
    """Return FCC structure."""
    axis = np.eye(3) * a
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    ).T
    n_atoms, types, elements = _get_structure_attrs(positions.shape[1], element_strings)

    fcc = PolymlpStructure(
        axis=axis,
        positions=positions,
        n_atoms=n_atoms,
        types=types,
        elements=elements,
    )
    return Prototype(fcc, "fcc", 52914, 4, (4, 4, 4))


def structure_bcc(element_strings: tuple, a: float = 4.0):
    """Return BCC structure."""
    axis = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).T
    n_atoms, types, elements = _get_structure_attrs(positions.shape[1], element_strings)

    fcc = PolymlpStructure(
        axis=axis,
        positions=positions,
        n_atoms=n_atoms,
        types=types,
        elements=elements,
    )
    return Prototype(fcc, "bcc", 76156, 2, (4, 4, 4))


def get_structure_type_element():
    """Return structure type."""
    structure_type = {
        104296: "Hg(LT)",
        105489: "FeB",
        108326: "Cr3Si",
        108682: "Pr(hP6)",
        109012: "Li(cI16)",
        109018: "Cs(HP)",
        109027: "Sr(HP)",
        109028: "Bi(I4/mcm)",
        109035: "Ca.15Sn.85",
        15535: "O2(hR2)",
        157920: "IrV",
        161381: "K(HP)",
        162242: "Ca(III)",
        162256: "Ga",
        165132: "B28",
        165725: "Co(tP28)",
        167204: "Ge(hP8)",
        168172: "I(Immm)",
        173517: "Ge(HP)",
        181082: "C(P1)",
        181908: "Si(I4/mmm)",
        182587: "Nickeline-NiAs",
        182732: "BN(P1)",
        182760: "C(C2/m)",
        185973: "C3N2",
        187431: "CaHg2",
        188336: "(Ca8)xCa2",
        189436: "B(tP50)",
        189460: "MnP",
        189786: "Si(oS16)",
        193439: "Graphite(2H)",
        194468: "I2",
        2091: "Se3S5",
        236662: "Sn",
        236858: "O8",
        245950: "Ge(cF136)",
        245956: "Ge",
        246446: "Bi(III)",
        248474: "Pu(oF8)",
        248500: "O2(mS4)",
        24892: "N2",
        26463: "S12",
        26482: "N2(cP8)",
        27249: "CO",
        280872: "Ta(tP30)",
        28540: "H2",
        30101: "Wurtzite-ZnS(2H)",
        30606: "Se(beta)",
        31170: "C(P63mc)",
        31692: "Pu(mP16)",
        31829: "Graphite(3R)",
        37090: "S6",
        41979: "Diamond-C(cF8)",
        42679: "Sb(mP4)",
        43058: "Mn(alpha)-Mn(cI58)",
        43211: "Po(alpha)",
        43216: "Sn(tI2)",
        43251: "S8(Fddd)",
        43539: "Ga(Cmcm)",
        52412: "Sc",
        52501: "Te(mP4)",
        52914: "ccp-Cu",
        52916: "La",
        56503: "GaSb",
        56897: "SmNiSb",
        57192: "Cs(tP8)",
        609832: "P(black)",
        616526: "As",
        62747: "B(hR12)",
        639810: "In",
        642937: "Mn(cP20)",
        644520: "N2(epsilon)",
        648333: "Pa",
        652633: "Sm",
        652876: "hcp-Mg",
        653045: "Se(gamma)",
        653048: "Te3",
        653381: "U",
        653719: "Bi",
        653797: "Pu(mS34)",
        656457: "Po(hR1)",
        76041: "U(beta)",
        76156: "bcc-W",
        76166: "U(beta)-CrFe",
        88815: "Graphite(oS16)",
        88820: "Si(HP)",
        97742: "Sb2Te3",
        # 42679-2; structure type: Sb(mP4)-2
        # 652876-2; structure type: hcp-Mg
        # 652876-3; structure type: hcp-Mg
    }
    return structure_type
