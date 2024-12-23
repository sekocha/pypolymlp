"""Functions for compressing vasprun.xml files."""

from xml.dom import minidom
from xml.etree import ElementTree as ET


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem)
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="", newl="")


def convert(vasprun):

    try:
        root = ET.parse(vasprun).getroot()
    except ET.ParseError:
        print("ET.ParseError:", vasprun)
        return False

    e = root.find("calculation").find("energy")
    f = root.find("calculation").find(".//*[@name='forces']")
    s = root.find("calculation").find(".//*[@name='stress']")
    st = root.find(".//*[@name='finalpos']")
    st2 = root.find(".//*[@name='atomtypes']")
    st3 = root.find(".//*[@name='atoms']")

    m1, c1 = ET.Element("modeling"), ET.Element("calculation")
    c1.append(e)
    c1.append(f)
    c1.append(s)
    c1.append(st)
    c1.append(st2)
    c1.append(st3)
    m1.append(c1)
    f = open(vasprun + ".polymlp", "w")
    print(prettify(m1), file=f)
    f.close()

    return True
