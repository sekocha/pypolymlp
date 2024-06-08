#!/usr/bin/env python
import argparse
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--vaspruns", nargs="*", type=str, help="vasprun.xml files")
    parser.add_argument("--n_jobs", type=int, default=1, help="number of parallel jobs")
    args = parser.parse_args()

    if args.n_jobs == 1:
        for vasp in args.vaspruns:
            convert(vasp)
    else:
        from joblib import Parallel, delayed

        res = Parallel(n_jobs=args.n_jobs)(
            delayed(convert)(vasp) for vasp in args.vaspruns
        )
