"""Functions for providing initial structures for systematic calculations."""

import numpy as np

from pypolymlp.calculator.auto.autocalc_utils import Prototype, get_atomic_size_scales
from pypolymlp.core.data_format import PolymlpStructure


def _get_structure_attrs(n_atom: int, element_strings: tuple):
    """Return structure attributes for elemental system."""
    n_atoms = [n_atom]
    types = np.zeros(n_atom, dtype=int)
    elements = [ele for n, ele in zip(n_atoms, element_strings) for _ in range(n)]
    return (n_atoms, types, elements)


def set_structure(axis: np.ndarray, positions: np.ndarray, element_strings: tuple):
    """Set PolymlpStructure instance."""
    n_atoms, types, elements = _get_structure_attrs(positions.shape[1], element_strings)
    st = PolymlpStructure(
        axis=axis,
        positions=positions,
        n_atoms=n_atoms,
        types=types,
        elements=elements,
    )
    return st


def get_structure_list_element(element_strings: tuple):
    """Return structure list for elemental systems."""
    scales = get_atomic_size_scales()
    scale = scales[element_strings[0]] * 1.1

    fcc = structure_fcc(element_strings, scale=scale)
    bcc = structure_bcc(element_strings, scale=scale)
    hcp = structure_hcp(element_strings, scale=scale)
    Sn_t = structure_Sn_tI2(element_strings, scale=scale)
    Sn = structure_Sn(element_strings, scale=scale)
    Bi = structure_Bi(element_strings, scale=scale)
    Sm = structure_Sm(element_strings, scale=scale)
    La = structure_La(element_strings, scale=scale)
    dia = structure_diamond(element_strings, scale=scale)
    sc = structure_sc(element_strings, scale=scale)
    black_P = structure_black_P(element_strings, scale=scale)
    As = structure_As(element_strings, scale=scale)
    Sb_m = structure_Sb_mP4(element_strings, scale=scale)
    Se = structure_gamma_Se(element_strings, scale=scale)
    Ga = structure_Ga(element_strings, scale=scale)
    Ga_cmcm = structure_Ga_Cmcm(element_strings, scale=scale)
    In = structure_In(element_strings, scale=scale)
    Si_i4 = structure_Si_I4mmm(element_strings, scale=scale)
    # Te = structure_Te(element_strings: tuple, scale: float = 1.3):
    return [
        fcc,
        bcc,
        hcp,
        Sn_t,
        Sn,
        Bi,
        Sm,
        La,
        dia,
        sc,
        black_P,
        As,
        Sb_m,
        Se,
        Ga,
        Ga_cmcm,
        In,
        Si_i4,
    ]


def structure_fcc(element_strings: tuple, scale: float = 1.3):
    """Return FCC structure."""
    a = 4.0 * scale
    axis = np.eye(3) * a
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "fcc", 52914, 4, (4, 4, 4))


def structure_bcc(element_strings: tuple, scale: float = 1.3):
    """Return BCC structure."""
    a = 3.2 * scale
    axis = np.eye(3) * a
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "bcc", 76156, 2, (4, 4, 4))


def structure_hcp(element_strings: tuple, scale: float = 1.3):
    """Return HCP structure."""
    axis = np.zeros((3, 3))
    a = 3.0 * scale
    axis[0, 0] = a
    axis[0, 1] = -a / 2.0
    axis[1, 1] = a * np.sqrt(3) / 2.0
    axis[2, 2] = a * np.sqrt(8.0 / 3.0)
    positions = np.array([[2 / 3, 1 / 3, 0.25], [1 / 3, 2 / 3, 0.75]]).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "hcp", 652876, 2, (4, 4, 4))


def structure_Sn_tI2(element_strings: tuple, scale: float = 1.3):
    """Return Sn(tI2) structure."""
    a = 3.4 * scale
    axis = np.eye(3) * a
    axis[2, 2] = a * 0.9
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Sn(tI2)", 43216, 2, (4, 4, 4))


def structure_Sn(element_strings: tuple, scale: float = 1.3):
    """Return Sn structure."""
    a = 5.0 * scale
    axis = np.eye(3) * a
    axis[2, 2] = a * 0.6
    positions = np.array(
        [[0.0, 0.0, 0.00], [0.0, 0.5, 0.25], [0.5, 0.5, 0.50], [0.5, 0.0, 0.75]]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Sn", 236662, 4, (4, 4, 4))


def structure_Bi(element_strings: tuple, scale: float = 1.3):
    """Return Bi structure."""
    axis = (
        scale
        * np.array(
            [
                [5.2857382131102684, 0.0000000000000000, -0.2974632175834797],
                [0.0000000000000000, 5.1179310158220206, 0.0000000000000000],
                [-1.0452934833151939, 0.0000000000000000, 2.6029704616435523],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.2384740361686681, 0.5, 0.8920306915057310],
            [0.7615259638313319, 0.5, 0.1079693084942690],
            [0.7384740361686681, 0.0, 0.8920306915057310],
            [0.2615259638313319, 0.0, 0.1079693084942690],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Bi", 653719, 4, (4, 4, 4))


def structure_Sm(element_strings: tuple, scale: float = 1.3):
    """Return Sm structure."""
    a = 3.0 * scale
    axis = np.zeros((3, 3))
    axis[0, 0] = a
    axis[0, 1] = -a / 2.0
    axis[1, 1] = a * np.sqrt(3) / 2.0
    axis[2, 2] = a * np.sqrt(8.0 / 3.0) * 4.5
    positions = np.array(
        [
            [0.0, 0.0, 0.0000000000000000],
            [2 / 3, 1 / 3, 0.1104635482669423],
            [0.0, 0.0, 0.2228697850663934],
            [2 / 3, 1 / 3, 0.3333333333333357],
            [1 / 3, 2 / 3, 0.4437968816002780],
            [2 / 3, 1 / 3, 0.5562031183997220],
            [1 / 3, 2 / 3, 0.6666666666666643],
            [0.0, 0.0, 0.7771302149336066],
            [1 / 3, 2 / 3, 0.8895364517330577],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Sm", 652633, 9, (6, 6, 1))


def structure_La(element_strings: tuple, scale: float = 1.3):
    """Return La structure."""
    a = 3.0 * scale
    axis = np.zeros((3, 3))
    axis[0, 0] = a
    axis[0, 1] = -a / 2.0
    axis[1, 1] = a * np.sqrt(3) / 2.0
    axis[2, 2] = a * np.sqrt(8.0 / 3.0) * 2
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [2 / 3, 1 / 3, 0.25], [1 / 3, 2 / 3, 0.75]]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "La", 52916, 4, (6, 6, 2))


def structure_diamond(element_strings: tuple, scale: float = 1.3):
    """Return diamond structure."""
    a = 5.5 * scale
    axis = np.eye(3) * a
    positions = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.00, 0.00, 0.00],
            [0.25, 0.75, 0.75],
            [0.00, 0.50, 0.50],
            [0.75, 0.25, 0.75],
            [0.50, 0.00, 0.50],
            [0.75, 0.75, 0.25],
            [0.50, 0.50, 0.00],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "diamond", 41979, 8, (4, 4, 4))


def structure_sc(element_strings: tuple, scale: float = 1.3):
    """Return SC structure."""
    a = 2.6 * scale
    axis = np.eye(3) * a
    positions = np.zeros((3, 1))
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "sc", 43211, 1, (4, 4, 4))


def structure_black_P(element_strings: tuple, scale: float = 1.3):
    """Return black-P structure."""
    axis = (
        scale
        * np.array(
            [
                [2.9375570911275237, 0.0, 0.0],
                [0.0, 9.5992513689171890, 0.0],
                [0.0, 0.0, 4.8915941941104810],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.0000000000000000, 0.6354685850174135, 0.4997888348867647],
            [0.5000000000000000, 0.8645314149825865, 0.5002111651132353],
            [0.0000000000000000, 0.8645314149825865, 0.9997888348867647],
            [0.5000000000000000, 0.6354685850174135, 0.0002111651132353],
            [0.5000000000000000, 0.1354685850174135, 0.4997888348867647],
            [0.0000000000000000, 0.3645314149825865, 0.5002111651132353],
            [0.5000000000000000, 0.3645314149825865, 0.9997888348867647],
            [0.0000000000000000, 0.1354685850174135, 0.0002111651132353],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "P(black)", 609832, 8, (6, 2, 4))


def structure_As(element_strings: tuple, scale: float = 1.3):
    """Return As structure."""
    axis = (
        scale
        * np.array(
            [
                [3.7042870595376334, 0.0000000000000001, 0.0000000000000000],
                [-1.8521435297688169, 3.2080066964695488, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 4.5299704829004872],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.0000000000000000, 0.0000000000000000, 0.5000000000000000],
            [0.6666666666666643, 0.3333333333333357, 0.8333333333333357],
            [0.3333333333333357, 0.6666666666666643, 0.1666666666666643],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "As", 616526, 3, (4, 4, 4))


def structure_Sb_mP4(element_strings: tuple, scale: float = 1.3):
    """Return Sb(mP4) structure."""
    axis = (
        scale
        * np.array(
            [
                [4.1229444681990444, 0.0000000000000000, 0.1797161542114127],
                [0.0000000000000000, 3.2746724677068730, 0.0000000000000000],
                [-0.1938952339603432, 0.0000000000000000, 5.2759819615741010],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.8535843302085411, 0.2500000000000000, 0.8277373769685283],
            [0.1464156697914873, 0.7500000000000000, 0.1722626230315143],
            [0.6456138768536945, 0.2500000000000000, 0.3278813991459302],
            [0.3543861231462486, 0.7500000000000000, 0.6721186008540982],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Sb(mP4)", 42679, 4, (4, 4, 4))


def structure_gamma_Se(element_strings: tuple, scale: float = 1.3):
    """Return gamma-Se structure."""
    axis = (
        scale
        * np.array(
            [
                [3.7027281480378562, -0.0000000000000001, 0.0000000000000000],
                [-1.8513640740189274, 3.2066566395084877, 0.0000000000000001],
                [0.0000000000000000, 0.0000000000000000, 4.5310395675849477],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.6675558007424769, 0.6675558007424769, 0.5000000000000000],
            [0.3324441992575231, 0.0000000000000000, 0.8333333333333357],
            [0.0000000000000000, 0.3324441992575231, 0.1666666666666643],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Se(gamma)", 653045, 3, (4, 4, 4))


def structure_Te(element_strings: tuple, scale: float = 1.3):
    """Return Te structure."""
    pass
    # TODO: Structure.
    # st = set_structure(axis, positions, element_strings)
    # return Prototype(st, "Te", 653048, 3, (4, 4, 4))


def structure_Ga(element_strings: tuple, scale: float = 1.3):
    """Return Ga structure."""
    axis = (
        scale
        * np.array(
            [
                [5.1605085524343917, 0.0000000000000000, 0.0000000000000000],
                [0.0000000000000000, 5.4114708370994267, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 4.9973441371133402],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.5000000000000000, 0.1473333539957338, 0.6990358352147297],
            [0.5000000000000000, 0.8526666460042662, 0.3009641647852703],
            [0.5000000000000000, 0.3526666460042662, 0.1990358352147297],
            [0.5000000000000000, 0.6473333539957338, 0.8009641647852703],
            [0.0000000000000000, 0.6473333539957338, 0.6990358352147297],
            [0.0000000000000000, 0.3526666460042662, 0.3009641647852703],
            [0.0000000000000000, 0.8526666460042662, 0.1990358352147297],
            [0.0000000000000000, 0.1473333539957338, 0.8009641647852703],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Ga", 162256, 8, (4, 4, 4))


def structure_Ga_Cmcm(element_strings: tuple, scale: float = 1.3):
    """Return Ga(Cmcm) structure."""
    axis = (
        scale
        * np.array(
            [
                [2.8228159836356812, 0.0000000000000000, 0.0000000000000000],
                [0.0000000000000000, 7.7042453365737842, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 3.2833198240463739],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.5000000000000000, 0.3579044935327715, 0.2500000000000000],
            [0.0000000000000000, 0.1420955064672285, 0.7500000000000000],
            [0.0000000000000000, 0.8579044935327715, 0.2500000000000000],
            [0.5000000000000000, 0.6420955064672285, 0.7500000000000000],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Ga(Cmcm)", 43539, 4, (6, 2, 6))


def structure_In(element_strings: tuple, scale: float = 1.3):
    """Return In structure."""
    axis = (
        scale
        * np.array(
            [
                [2.8870441679037220, 0.0000000000000000, 0.0000000000000000],
                [0.0000000000000000, 2.8870441679037220, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 4.1016577357079225],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.0000000000000000, 0.0000000000000000, 0.0000000000000000],
            [0.5000000000000000, 0.5000000000000000, 0.5000000000000000],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "In", 639810, 2, (4, 4, 4))


def structure_Si_I4mmm(element_strings: tuple, scale: float = 1.3):
    """Return Si(I4mm) structure."""
    axis = (
        scale
        * np.array(
            [
                [6.7683690936373777, 0.0000000000000000, 0.0000000000000000],
                [0.0000000000000000, 6.7683690936373777, 0.0000000000000000],
                [0.0000000000000000, 0.0000000000000000, 4.0195736763805785],
            ]
        ).T
    )
    positions = np.array(
        [
            [0.8274296477599776, 0.1725703522400224, 0.0000000000000000],
            [0.6725703522400224, 0.3274296477599776, 0.5000000000000000],
            [0.3274296477599776, 0.3274296477599776, 0.5000000000000000],
            [0.1725703522400224, 0.1725703522400224, 0.0000000000000000],
            [0.3274296477599776, 0.6725703522400224, 0.5000000000000000],
            [0.1725703522400224, 0.8274296477599776, 0.0000000000000000],
            [0.8274296477599776, 0.8274296477599776, 0.0000000000000000],
            [0.6725703522400224, 0.6725703522400224, 0.5000000000000000],
        ]
    ).T
    st = set_structure(axis, positions, element_strings)
    return Prototype(st, "Si(I4mm)", 181908, 8, (4, 4, 4))


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
