#!/usr/bin/env python
import time

import numpy as np
import phono3py
import phonopy
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc import Symfc

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization_sym import MinimizeSym
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.displacements import (
    generate_random_const_displacements,
    get_structures_from_displacements,
)
from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    phonopy_supercell,
    structure_to_phonopy_cell,
)


class PolymlpFC:

    def __init__(
        self,
        supercell=None,
        phono3py_yaml=None,
        use_phonon_dataset=False,
        pot=None,
        params=None,
        coeffs=None,
        properties=None,
        cutoff_fc3=None,
        cutoff_fc4=None,
    ):
        """
        Parameters
        ----------
        supercell: Supercell in phonopy format or structure dict
        pot, (params_dict and coeffs), or Properties object: polynomal MLP
        """

        if pot is None and params is None and properties is None:
            self.prop = None
        else:
            if properties is not None:
                self.prop = properties
            else:
                self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self.__initialize_supercell(
            supercell=supercell,
            phono3py_yaml=phono3py_yaml,
            use_phonon_dataset=use_phonon_dataset,
        )
        """
        np.set_printoptions(precision=15)
        print(supercell.cell)
        print(supercell.scaled_positions)
        print(supercell.numbers)
        """
        self.__fc2 = None
        self.__fc3 = None
        self.__fc4 = None
        self.__disps = None
        self.__forces = None

        if cutoff_fc3 is not None:
            self.cutoff_fc3 = cutoff_fc3
        else:
            self.__cutoff_fc3 = None

        if cutoff_fc4 is not None:
            self.cutoff_fc4 = cutoff_fc4
        else:
            self.__cutoff_fc4 = None

    def __initialize_supercell(
        self, supercell=None, phono3py_yaml=None, use_phonon_dataset=False
    ):

        if supercell is not None:
            if isinstance(supercell, PolymlpStructure):
                self.__supercell_dict = supercell
                self.__supercell_ph = structure_to_phonopy_cell(supercell)
            elif isinstance(supercell, phonopy.structure.cells.Supercell):
                self.__supercell_dict = phonopy_cell_to_structure(supercell)
                self.__supercell_ph = supercell
            else:
                raise ValueError(
                    "PolymlpFC: type(supercell) must be" " dict or phonopy supercell"
                )

        elif phono3py_yaml is not None:
            print("Supercell is read from:", phono3py_yaml)
            (self.__supercell_ph, self.__disps, self.__st_dicts) = (
                parse_phono3py_yaml_fcs(
                    phono3py_yaml, use_phonon_dataset=use_phonon_dataset
                )
            )
            self.__supercell_dict = phonopy_cell_to_structure(self.__supercell_ph)

        else:
            raise ValueError(
                "PolymlpFC: supercell or phonon3py_yaml"
                " is required for initialization"
            )

        self.__N = len(self.__supercell_ph.symbols)
        return self

    def sample(self, n_samples=100, displacements=0.001, is_plusminus=False):

        self.__disps, self.__st_dicts = generate_random_const_displacements(
            self.__supercell_dict,
            n_samples=n_samples,
            displacements=displacements,
            is_plusminus=is_plusminus,
        )
        return self

    def run_geometry_optimization(self, gtol=1e-5, method="CG"):

        print("Running geometry optimization")
        try:
            minobj = MinimizeSym(self.__supercell_dict, properties=self.prop)
        except ValueError:
            print("No geomerty optimization is performed.")
            return self

        minobj.run(gtol=gtol, method=method)
        print("Residual forces:")
        print(minobj.residual_forces.T)
        print("E0:", minobj.energy)
        print("n_iter:", minobj.n_iter)
        print("Fractional coordinate changes:")
        diff_positions = self.__supercell_dict.positions - minobj.structure.positions
        print(diff_positions.T)
        print("Success:", minobj.success)

        if minobj.success:
            self.__supercell_dict = minobj.structure
            self.__supercell_ph = structure_to_phonopy_cell(self.__supercell_dict)
            if self.__disps is not None:
                self.displacements = self.__disps

        return self

    def set_cutoff(self, cutoff=7.0):
        self.cutoff = cutoff
        return self

    def __compute_forces(self):

        _, forces, _ = self.prop.eval_multiple(self.__st_dicts)
        _, residual_forces, _ = self.prop.eval(self.__supercell_dict)
        for f in forces:
            f -= residual_forces
        return forces

    def run_fc2(self):
        """Construct fc2 basis and solve FC2"""

        symfc = Symfc(
            self.__supercell_ph,
            displacements=self.__disps.transpose((0, 2, 1)),
            forces=self.__forces.transpose((0, 2, 1)),
        ).run(2)
        self.__fc2 = symfc.force_constants[2]

        return self

    def run_fc2fc3(self, batch_size=100, sum_rule_basis=True):
        """Construct fc2 + fc3 basis and solve FC2 + FC3

        disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        """
        N = self.__forces.shape[2]
        use_mkl = False if self.__cutoff is None and N > 400 else True
        symfc = Symfc(
            self.__supercell_ph,
            displacements=self.__disps.transpose((0, 2, 1)),
            forces=self.__forces.transpose((0, 2, 1)),
            cutoff=self.__cutoff,
            use_mkl=use_mkl,
            log_level=1,
        ).run(3, batch_size=batch_size)
        self.__fc2, self.__fc3 = symfc.force_constants[2], symfc.force_constants[3]

        # disps = self.__disps.transpose((0, 2, 1))
        # forces = self.__forces.transpose((0, 2, 1))

        """
        from pypolymlp.symfc.dev.symfc_basis_dev import run_basis_fc2
        compress_mat_fc2, compress_eigvecs_fc2, atomic_decompr_idx_fc2 = run_basis_fc2(
            self.__supercell_ph,
            fc_cutoff=None,
        )
        """

        """
        from symfc.basis_sets.basis_sets_O2 import FCBasisSetO2
        from symfc.basis_sets.basis_sets_O3 import FCBasisSetO3
        from symfc.solvers.solver_O2O3 import FCSolverO2O3

        fc2_basis = FCBasisSetO2(self.__supercell_ph, use_mkl=False).run()
        fc3_basis = FCBasisSetO3(
            self.__supercell_ph, cutoff=self.__cutoff, use_mkl=True, log_level=1,
        ).run()
        fc_solver = FCSolverO2O3([fc2_basis, fc3_basis], use_mkl=True, log_level=1)
        fc_solver.solve(disps, forces, batch_size=batch_size)
        self.__fc2, self.__fc3 = fc_solver.compact_fc
        """
        return self

    def run_fc2fc3fc4(self, batch_size=100):
        """Construct fc2 + fc3 + fc4 basis and solve FC2 + FC3 + FC4"""

        """
        disps: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        forces: (n_str, 3, n_atom) --> (n_str, n_atom, 3)
        """
        cutoff = {3: self.__cutoff_fc3, 4: self.__cutoff_fc4}
        symfc = Symfc(
            self.__supercell_ph,
            displacements=self.__disps.transpose((0, 2, 1)),
            forces=self.__forces.transpose((0, 2, 1)),
            cutoff=cutoff,
            use_mkl=True,
            log_level=1,
        )
        symfc.run(4, batch_size=batch_size)

        self.__fc2 = symfc.force_constants[2]
        self.__fc3 = symfc.force_constants[3]
        self.__fc4 = symfc.force_constants[4]

        """
        n_a_compress_mat = symfc.basis_set[3].compact_compression_matrix
        basis_set = symfc.basis_set[3].basis_set
        compact_basis = n_a_compress_mat @ basis_set
        print(basis_set.shape)
        print(np.linalg.norm(compact_basis) ** 2)

        n_a_compress_mat = symfc.basis_set[4].compact_compression_matrix
        basis_set = symfc.basis_set[4].basis_set
        compact_basis = n_a_compress_mat @ basis_set
        print(basis_set.shape)
        print(np.linalg.norm(compact_basis) ** 2)
        """

        return self

    def run(
        self,
        disps=None,
        forces=None,
        batch_size=100,
        sum_rule_basis=True,
        write_fc=True,
        only_fc2=False,
        only_fc2fc3=False,
    ):

        if disps is not None:
            self.displacements = disps

        if forces is None:
            print("Computing forces using polymlp")
            t1 = time.time()
            self.forces = np.array(self.__compute_forces())
            t2 = time.time()
            print(" elapsed time (computing forces) =", t2 - t1)
        else:
            self.forces = forces

        if only_fc2:
            self.run_fc2()
        elif only_fc2fc3:
            self.run_fc2fc3(batch_size=batch_size, sum_rule_basis=sum_rule_basis)
        else:
            self.run_fc2fc3fc4(batch_size=batch_size)

        if write_fc:
            if self.__fc2 is not None:
                print("Writing fc2.hdf5")
                write_fc2_to_hdf5(self.__fc2)
            if self.__fc3 is not None:
                print("Writing fc3.hdf5")
                write_fc3_to_hdf5(self.__fc3)
            if self.__fc4 is not None:
                print("No function to write fc4.hdf5")

        return self

    @property
    def displacements(self):
        return self.__disps

    @property
    def forces(self):
        return self.__forces

    @property
    def structures(self):
        return self.__st_dicts

    @displacements.setter
    def displacements(self, disps):
        """disps: Displacements (n_str, 3, n_atom)"""
        if not disps.shape[1] == 3 or not disps.shape[2] == self.__N:
            raise ValueError("displacements must have a shape of " "(n_str, 3, n_atom)")
        self.__disps = disps
        self.__st_dicts = get_structures_from_displacements(
            self.__disps, self.__supercell_dict
        )

    @forces.setter
    def forces(self, f):
        """forces: shape=(n_str, 3, n_atom)"""
        if not f.shape[1] == 3 or not f.shape[2] == self.__N:
            raise ValueError("forces must have a shape of " "(n_str, 3, n_atom)")
        self.__forces = f

    @structures.setter
    def structures(self, st_dicts):
        self.__st_dicts = st_dicts

    @property
    def fc2(self):
        return self.__fc2

    @property
    def fc3(self):
        return self.__fc3

    @property
    def fc4(self):
        return self.__fc4

    @property
    def supercell_phonopy(self):
        return self.__supercell_ph

    @property
    def supercell_dict(self):
        return self.__supercell_dict

    @property
    def cutoff_fc3(self):
        return self.__cutoff_fc3

    @property
    def cutoff_fc4(self):
        return self.__cutoff_fc4

    @cutoff_fc3.setter
    def cutoff_fc3(self, value):
        print("Cutoff radius (FC3):", value, "(ang.)")
        self.__cutoff_fc3 = value
        return self

    @cutoff_fc4.setter
    def cutoff_fc4(self, value):
        print("Cutoff radius (FC4):", value, "(ang.)")
        self.__cutoff_fc4 = value
        return self


if __name__ == "__main__":

    import argparse
    import signal

    from pypolymlp.core.interface_vasp import Poscar

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

    parser.add_argument("--pot", type=str, default=None, help="polymlp file")
    parser.add_argument(
        "--fc_n_samples",
        type=int,
        default=None,
        help="Number of random displacement samples",
    )
    parser.add_argument(
        "--disp",
        type=float,
        default=0.03,
        help="Displacement (in Angstrom)",
    )
    parser.add_argument(
        "--is_plusminus",
        action="store_true",
        help="Plus-minus displacements will be generated.",
    )
    parser.add_argument(
        "--geometry_optimization",
        action="store_true",
        help="Geometry optimization is performed " "for initial structure.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for FC solver.",
    )
    parser.add_argument(
        "--cutoff_fc3",
        type=float,
        default=None,
        help="Cutoff radius for setting zero elements (FC3).",
    )
    parser.add_argument(
        "--cutoff_fc4",
        type=float,
        default=None,
        help="Cutoff radius for setting zero elements (FC4).",
    )
    parser.add_argument("--run_ltc", action="store_true", help="Run LTC calculations")
    parser.add_argument(
        "--ltc_mesh",
        type=int,
        nargs=3,
        default=[19, 19, 19],
        help="k-mesh used for phono3py calculation",
    )
    args = parser.parse_args()

    unitcell_dict = Poscar(args.poscar).structure
    supercell_matrix = np.diag(args.supercell)
    supercell = phonopy_supercell(unitcell_dict, supercell_matrix)

    polyfc = PolymlpFC(
        supercell=supercell,
        pot=args.pot,
        cutoff_fc3=args.cutoff_fc3,
        cutoff_fc4=args.cutoff_fc4,
    )

    if args.fc_n_samples is not None:
        polyfc.sample(
            n_samples=args.fc_n_samples,
            displacements=args.disp,
            is_plusminus=args.is_plusminus,
        )

    if args.geometry_optimization:
        polyfc.run_geometry_optimization()

    polyfc.run(write_fc=True, batch_size=args.batch_size)

    if args.run_ltc:
        ph3 = phono3py.load(
            unitcell_filename=args.poscar,
            supercell_matrix=supercell_matrix,
            primitive_matrix="auto",
            log_level=1,
        )
        ph3.mesh_numbers = args.ltc_mesh
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(temperatures=range(0, 1001, 10), write_kappa=True)
