"""IO functions for modulation."""

import os

from pypolymlp.utils.vasp_utils import write_poscar_file

from .modulation_utils import ModulationNode


def save_nodes_to_file(
    nodes: list[ModulationNode], filename: str = "polymlp_modulation.yaml"
):
    """Save attributes of nodes to a file."""
    os.makedirs("polymlp_modulation_poscars", exist_ok=True)

    idmap = {}
    idx = 1
    for nodes_depth in nodes:
        for node in nodes_depth:
            idmap[(node.depth, node.parent_id, node.id)] = idx
            idx += 1

    with open(filename, "w") as f:
        print("nodes:", file=f)
        idx = 1
        for depth, nodes_depth in enumerate(nodes):
            if len(nodes_depth) == 0:
                continue
            print("- depth:", depth, file=f)
            for node in nodes_depth:
                parent_id = 0 if node.parent_id is None else node.parent_id
                print("  - node_id:        ", idx, file=f)
                print("    spacegroup:     ", node.spacegroup, file=f)
                print("    enthalpy:       ", node.enthalpy, file=f)
                print("    parent_id:      ", node.parent_id, file=f)
                print("    id:             ", node.id, file=f)
                print("    is_imaginary:   ", node.is_imaginary, file=f)
                if node.equivalent_shallower is not None:
                    key = tuple(node.equivalent_shallower)
                    print("    equivalent_node:", idmap[key], file=f)

                poscarfile = "polymlp_modulation_poscars/poscar-" + str(idx).zfill(4)
                header = (
                    "modulation-"
                    + "-".join([str(node.depth), str(parent_id), str(node.id)])
                    + ", Spacegroup: "
                    + node.spacegroup
                    + ", Imaginary: "
                    + str(node.is_imaginary)
                )
                write_poscar_file(node.structure, filename=poscarfile, header=header)
                idx += 1
            print(file=f)
