"""Run symfc."""

from symfc import Symfc
from symfc.utils.utils import SymfcAtoms

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.structure_utils import supercell_diagonal

if __name__ == "__main__":

    import argparse
    import signal
    import time

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--poscar", type=str, default=None, help="poscar")
    parser.add_argument(
        "--supercell",
        nargs=3,
        type=int,
        default=None,
        help="Supercell size (diagonal components)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="FC order.",
    )
    parser.add_argument(
        "--cutoff_fc3",
        type=float,
        default=None,
        help="Cutoff radius for FC3.",
    )
    parser.add_argument(
        "--cutoff_fc4",
        type=float,
        default=None,
        help="Cutoff radius for FC4.",
    )
    args = parser.parse_args()

    """Phonopy use
    unitcell = Poscar(args.poscar).structure
    supercell = phonopy_supercell(unitcell, np.diag(args.supercell))
    """
    unitcell = Poscar(args.poscar).structure
    supercell = supercell_diagonal(unitcell, args.supercell)
    supercell = SymfcAtoms(
        numbers=supercell.types,
        cell=supercell.axis.T,
        scaled_positions=supercell.positions.T,
    )
    cutoff = {3: args.cutoff_fc3, 4: args.cutoff_fc4}

    t1 = time.time()
    symfc = Symfc(supercell, use_mkl=True, log_level=1, cutoff=cutoff)
    symfc.compute_basis_set(args.order)
    t2 = time.time()
    print("Max FC order:", args.order)
    print("Elapsed time (Basis sets):", "{:.3f}".format(t2 - t1))
    for order in range(2, args.order + 1):
        print(
            "Number of basis vectors (order " + str(order) + "):",
            symfc.basis_set[order].basis_set.shape[1],
        )
