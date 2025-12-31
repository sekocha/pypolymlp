"""Functions for providing initial binary structures for systematic calculations."""

# import numpy as np
#
# from pypolymlp.calculator.auto.dataclass import Prototype
# from pypolymlp.core.data_format import PolymlpStructure

# from pypolymlp.calculator.auto.structure_element import set_structure


# def get_structure_list_binary(element_strings: tuple):
#     """Return structure list for binary alloy systems."""
#     pass
#
#
# def structure_MoNi4_D1a_x020(element_strings: tuple):
#     """Return MoNi4(D1a)(x=0.20) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "MoNi4(D1a)(x=0.20)", "107998-01", 10, (2, 2, 2))
#
#
# def structure_MoNi4_D1a_x080(element_strings: tuple):
#     """Return MoNi4(D1a)(x=0.80) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "MoNi4(D1a)(x=0.80)", "107998-10", 10, (2, 2, 2))
#
#
# def structure_Ni3Sn_D019_x025(element_strings: tuple):
#     """Return Ni3Sn(D019)(x=0.25) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Ni3Sn(D019)(x=0.25)", "104506-10", 16, (2, 2, 2))
#
#
# def structure_Ni3Sn_D019_x075(element_strings: tuple):
#     """Return Ni3Sn(D019)(x=0.75) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Ni3Sn(D019)(x=0.75)", "104506-01", 16, (2, 2, 2))
#
#
# def structure_Ni3Ti_D024_x025(element_strings: tuple):
#     """Return Ni3Ti(D024)(x=0.25) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Ni3Ti(D024)(x=0.25)", "649037-10", 16, (2, 2, 2))
#
#
# def structure_Ni3Ti_D024_x075(element_strings: tuple):
#     """Return Ni3Ti(D024)(x=0.75) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Ni3Ti(D024)(x=0.75)", "649037-01", 16, (2, 2, 2))
#
#
# def structure_AuCu3_L12_x025(element_strings: tuple):
#     """Return AuCu3(L12)(x=0.25) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "AuCu3(L12)(x=0.25)", "181127-01", 4, (4, 4, 4))
#
#
# def structure_AuCu3_L12_x075(element_strings: tuple):
#     """Return AuCu3(L12)(x=0.75) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "AuCu3(L12)(x=0.75)", "181127-10", 4, (4, 4, 4))
#
#
# def structure_Al3Zr_D023_x025(element_strings: tuple):
#     """Return Al3Zr(D023)(x=0.25) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Al3Zr(D023)(x=0.25)", "416747-10", 16, (2, 2, 2))
#
#
# def structure_Al3Zr_D023_x075(element_strings: tuple):
#     """Return Al3Zr(D023)(x=0.75) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Al3Zr(D023)(x=0.75)", "416747-01", 16, (2, 2, 2))
#
#
# def structure_Al3Ti_D022_x025(element_strings: tuple):
#     """Return Al3Ti(D022)(x=0.25) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Al3Ti(D022)(x=0.25)", "105191-10", 8, (2, 2, 2))
#
#
# def structure_Al3Ti_D022_x075(element_strings: tuple):
#     """Return Al3Ti(D022)(x=0.75) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Al3Ti(D022)(x=0.75)", "105191-01", 8, (2, 2, 2))
#
#
# def structure_AlCu2Mn_L21_x025(element_strings: tuple):
#     """Return AlCu2Mn(L21)(x=0.25) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "AlCu2Mn(L21)(x=0.25)", "188260-01", 16, (2, 2, 2))
#
#
# def structure_AlCu2Mn_L21_x075(element_strings: tuple):
#     """Return AlCu2Mn(L21)(x=0.75) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "AlCu2Mn(L21)(x=0.75)", "188260-10", 16, (2, 2, 2))
#
#
# def structure_InNi2_B82_x033(element_strings: tuple):
#     """Return InNi2(B82)(x=0.33) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "InNi2(B82)(x=0.33)", "105948-10", 6, (2, 2, 2))
#
#
# def structure_InNi2_B82_x067(element_strings: tuple):
#     """Return InNi2(B82)(x=0.67) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "InNi2(B82)(x=0.67)", "105948-01", 6, (2, 2, 2))
#
#
# def structure_Fe2P_C22_x033(element_strings: tuple):
#     """Return Fe2P(C22)(x=0.33) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Fe2P(C22)(x=0.33)", "611176-10", 18, (2, 2, 2))
#
#
# def structure_Fe2P_C22_x067(element_strings: tuple):
#     """Return Fe2P(C22)(x=0.67) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Fe2P(C22)(x=0.67)", "611176-01", 18, (2, 2, 2))
#
#
# def structure_CrSi2_C40_x033(element_strings: tuple):
#     """Return CrSi2(C40)(x=0.33) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "CrSi2(C40)(x=0.33)", "16504-10", 9, (2, 2, 2))
#
#
# def structure_CrSi2_C40_x067(element_strings: tuple):
#     """Return CrSi2(C40)(x=0.67) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "CrSi2(C40)(x=0.67)", "16504-01", 9, (2, 2, 2))
#
#
# def structure_Cu2Sb_C38_x033(element_strings: tuple):
#     """Return Cu2Sb(C38)(x=0.33) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Cu2Sb(C38)(x=0.33)", "610464-01", 6, (2, 2, 2))
#
#
# def structure_Cu2Sb_C38_x067(element_strings: tuple):
#     """Return Cu2Sb(C38)(x=0.67) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Cu2Sb(C38)(x=0.67)", "610464-10", 6, (2, 2, 2))
#
#
# def structure_MgZn2_C14_x033(element_strings: tuple):
#     """Return MgZn2(C14)(x=0.33) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "MgZn2(C14)(x=0.33)", "625334-10", 12, (2, 2, 2))
#
#
# def structure_MgZn2_C14_x067(element_strings: tuple):
#     """Return MgZn2(C14)(x=0.67) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "MgZn2(C14)(x=0.67)", "625334-01", 12, (2, 2, 2))
#
#
# def structure_Si2U3_D5a_x040(element_strings: tuple):
#     """Return Si2U3(D5a)(x=0.40) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Si2U3(D5a)(x=0.40)", "639227-01", 10, (2, 2, 2))
#
#
# def structure_Si2U3_D5a_x060(element_strings: tuple):
#     """Return Si2U3(D5a)(x=0.60) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "Si2U3(D5a)(x=0.60)", "639227-10", 10, (2, 2, 2))
#
#
# def structure_AuCu_L10_x050(element_strings: tuple):
#     """Return AuCu(L10)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "AuCu(L10)(x=0.50)", "59508-01", 2, (4, 4, 4))
#
#
# def structure_CoU_Ba_x050(element_strings: tuple):
#     """Return CoU(Ba)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "CoU(Ba)(x=0.50)", "102712-01", 16, (2, 2, 2))
#
#
# def structure_CsCl_B2_x050(element_strings: tuple):
#     """Return CsCl(B2)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "CsCl(B2)(x=0.50)", "650527-01", 2, (4, 4, 4))
#
#
# def structure_NiAs_B81_x050(element_strings: tuple):
#     """Return NiAs(B81)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "NiAs(B81)(x=0.50)", "626692-01", 4, (4, 4, 4))
#
#
# def structure_WC_Bh_x050(element_strings: tuple):
#     """Return WC(Bh)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "WC(Bh)(x=0.50)", "644708-01", 2, (4, 4, 4))
#
#
# def structure_FeSi_B20_x050(element_strings: tuple):
#     """Return FeSi(B20)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "FeSi(B20)(x=0.50)", "635060-01", 8, (2, 2, 2))
#
#
# def structure_NaTl_B32_x050(element_strings: tuple):
#     """Return NaTl(B32)(x=0.50) structure."""
#     st = set_structure(axis, positions, element_strings)
#     return Prototype(st, "NaTl(B32)(x=0.50)", "103775-01", 16, (2, 2, 2))
