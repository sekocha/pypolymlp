"""Functions for compressing vasprun.xml files."""

from xml.dom import minidom
from xml.etree import ElementTree as ET

from pypolymlp.core.interface_vasp import check_vasprun_type


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem)
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="", newl="")


def compress_vaspruns(vasprun):
    """Compress vasprun.xml for single point calculation to a file."""
    try:
        root = ET.parse(vasprun).getroot()
    except ET.ParseError:
        print("ET.ParseError:", vasprun)
        return False

    md, _ = check_vasprun_type(root=root)
    if md:
        raise RuntimeError("Do not compress vasprun.xml from MD calculation.")

    e = root.find("calculation").find("energy")
    f = root.find("calculation").find(".//*[@name='forces']")
    s = root.find("calculation").find(".//*[@name='stress']")
    st = root.find(".//*[@name='finalpos']")
    st2 = root.find(".//*[@name='atomtypes']")
    st3 = root.find(".//*[@name='atoms']")

    m1, c1 = ET.Element("modeling"), ET.Element("calculation")
    c1.extend([e, f, s, st, st2, st3])
    m1.append(c1)

    with open(vasprun + ".polymlp", "w") as f:
        print(prettify(m1), file=f)

    return True


def compress_vaspruns_md(vasprun):
    """Compress vasprun.xml for MD calculation to a file."""
    try:
        root = ET.parse(vasprun).getroot()
    except ET.ParseError:
        print("ET.ParseError:", vasprun)
        return False

    m1 = ET.Element("modeling")
    tag = root.find(".//*[@name='IBRION']")
    st2 = root.find(".//*[@name='atomtypes']")
    st3 = root.find(".//*[@name='atoms']")
    m1.extend([tag, st2, st3])

    cals = root.findall("calculation")
    for cal in cals:
        c1 = ET.Element("calculation")
        e = cal.find("energy")
        f = cal.find(".//*[@name='forces']")
        s = cal.find(".//*[@name='stress']")
        st = cal.find("structure")
        c1.extend([e, f, s, st])
        m1.append(c1)

    with open(vasprun + ".polymlp", "w") as f:
        print(prettify(m1), file=f)

    return True
