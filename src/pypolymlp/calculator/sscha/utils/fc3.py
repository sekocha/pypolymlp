"""Function for computing third order force constants in post process."""

import argparse
import signal

import numpy as np
import phono3py
from phono3py.file_IO import write_fc3_to_hdf5
from phono3py.other.kaccum import KappaDOSTHM
from phono3py.phonon.grid import get_ir_grid_points

from pypolymlp.calculator.fc import PolymlpFC
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.sscha.harmonic_real import HarmonicReal
from pypolymlp.calculator.sscha.sscha_utils import Restart


def write_dos_from_frequencies(
    frequencies,
    bzgrid,
    ir_grid_points=None,
    ir_grid_weights=None,
    ir_grid_map=None,
    num_sampling_points=201,
    filename="total_dos.dat",
):

    if ir_grid_points is None:
        n_ir_grid_points = frequencies.shape[0]
        ir_grid_points = np.arange(n_ir_grid_points, dtype=int)
        ir_grid_weights = np.ones(n_ir_grid_points)
        ir_grid_map = np.arange(n_ir_grid_points, dtype=int)

    kappados = KappaDOSTHM(
        np.ones(frequencies.shape)[None, :, :, None],
        frequencies,
        bzgrid,
        ir_grid_points,
        ir_grid_weights=ir_grid_weights,
        ir_grid_map=ir_grid_map,
        num_sampling_points=num_sampling_points,
    )
    x, y = kappados.get_kdos()
    dos = np.vstack([x, y[0, :, 1, 0]]).T
    np.savetxt(filename, dos)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="sscha_results.yaml",
        help="sscha_results.yaml file to be parsed.",
    )
    parser.add_argument(
        "--fc2",
        type=str,
        default="fc2.hdf5",
        help="fc2.hdf5 file to be parsed.",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=None,
        help="polymlp.lammps file",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=10000,
        help="Number of sample supercells",
    )
    parser.add_argument(
        "--mesh",
        type=int,
        nargs=3,
        default=[19, 19, 19],
        help="k-mesh used for phono3py calculation",
    )
    args = parser.parse_args()

    res = Restart(args.yaml, fc2hdf5=args.fc2)

    pot = res.polymlp if args.pot is None else args.pot
    prop = Properties(pot=pot)

    print("Restart: SSCHA FC3 calculation")
    print("  yaml:        ", args.yaml)
    print("  fc2 :        ", args.fc2)
    print("  mlp :        ", pot)
    print("  n_structures:", args.n_samples)
    print("  temperature :", res.temperature)

    ph_real = HarmonicReal(
        res.supercell,
        prop,
        n_unitcells=res.n_unitcells,
        fc2=res.force_constants,
    )
    ph_real.run(t=res.temperature, n_samples=args.n_samples)
    """Codes up to here are totally the same as used in distribution.py"""

    polyfc = PolymlpFC(supercell=res.supercell_phonopy)
    polyfc.run(disps=ph_real.displacements, forces=ph_real.forces, write_fc=False)

    print("Writing fc3.hdf5")
    write_fc3_to_hdf5(polyfc.fc3)

    print("Running phono3py to compute real self energy.")
    ph3 = phono3py.load(
        unitcell=res.unitcell_phonopy,
        supercell_matrix=res.supercell_matrix,
        primitive_matrix="auto",
        log_level=1,
        is_mesh_symmetry=True,
    )
    ph3.mesh_numbers = args.mesh
    ph3.init_phph_interaction()
    ph3.run_phonon_solver()

    """ freq: shape=(n_grg, n_band)"""
    freq, _, _ = ph3.get_phonon_data()
    bz_grid = ph3._bz_grid

    ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(bz_grid)

    ir_grid_points_bzg = bz_grid.grg2bzg[ir_grid_points]
    ir_grid_map_bzg = bz_grid.grg2bzg[ir_grid_map]
    freq_ir = freq[ir_grid_points_bzg]

    print("Max freq = ", np.max(freq_ir))
    write_dos_from_frequencies(
        freq_ir,
        bz_grid,
        # ir_grid_points = ir_grid_points_bzg,
        # ir_grid_weights = ir_grid_weights,
        # ir_grid_map = ir_grid_map_bzg,
        ir_grid_points=ir_grid_points,
        ir_grid_weights=ir_grid_weights,
        ir_grid_map=ir_grid_map,
        filename="total_dos_fc2.dat",
    )

    print("Max freq = ", np.max(freq_ir))
    write_dos_from_frequencies(
        freq[bz_grid.grg2bzg],
        bz_grid,
        filename="total_dos_fc2_no_ir.dat",
    )

#    frequency_points, deltas = ph3.run_real_self_energy(
#        grid_points=ir_grid_points_bzg,
#        temperatures=[res.temperature],
#        frequency_points_at_bands=True,
#    )
#    deltas = deltas[0][0]
#    freq_ir_shifted = freq_ir + deltas
#
#    write_dos_from_frequencies(
#        freq_ir_shifted,
#        bz_grid,
#        ir_grid_points = ir_grid_points,
#        ir_grid_weights = ir_grid_weights,
#        ir_grid_map = ir_grid_map,
#        # ir_grid_points = ir_grid_points_bzg,
#        # ir_grid_weights = ir_grid_weights,
#        # ir_grid_map = ir_grid_map_bzg,
#        filename="total_dos_shifted.dat",
#    )
#
