"""Class for calculating FCs using polymlp."""

import time
from typing import Literal, Optional, Union

import numpy as np
import phonopy
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc import Symfc
from symfc.utils.cutoff_tools import FCCutoff

from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.str_opt.optimization import GeometryOptimization
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.displacements import (
    generate_random_const_displacements,
    get_structures_from_displacements,
)
from pypolymlp.core.interface_phono3py import parse_phono3py_yaml_fcs
from pypolymlp.utils.phonopy_utils import (
    phonopy_cell_to_structure,
    structure_to_phonopy_cell,
)


class PolymlpFC:
    """Class for calculating FCs using polymlp."""

    def __init__(
        self,
        supercell: Optional[PolymlpStructure] = None,
        phono3py_yaml: Optional[str] = None,
        use_phonon_dataset: bool = False,
        pot: Optional[str] = None,
        params: Optional[Union[PolymlpParams, list[PolymlpParams]]] = None,
        coeffs: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        properties: Optional[Properties] = None,
        cutoff: float = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        supercell: Supercell in PolymlpStructure or phonopy format.
        phono3py_yaml: phono3py.yaml file.
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.
        cutoff: Cutoff radius in angstroms.

        Any one of supercell and phono3py_yaml is needed.
        Any one of pot, (params, coeffs), and properties is needed.
        """

        if pot is None and params is None and properties is None:
            self.prop = None
        else:
            if properties is not None:
                self.prop = properties
            else:
                self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._initialize_supercell(
            supercell=supercell,
            phono3py_yaml=phono3py_yaml,
            use_phonon_dataset=use_phonon_dataset,
        )
        self._fc2 = None
        self._fc3 = None
        self._fc4 = None
        self._symfc = None
        self._disps = None
        self._forces = None

        if cutoff is not None:
            self.cutoff = cutoff
        else:
            self._cutoff = None
            self._fc_cutoff = None

        self._verbose = verbose

    def _initialize_supercell(
        self, supercell=None, phono3py_yaml=None, use_phonon_dataset=False
    ):

        if supercell is not None:
            if isinstance(supercell, PolymlpStructure):
                self._supercell = supercell
                self._supercell_ph = structure_to_phonopy_cell(supercell)
            elif isinstance(supercell, phonopy.structure.cells.Supercell):
                self._supercell = phonopy_cell_to_structure(supercell)
                self._supercell_ph = supercell
            else:
                raise ValueError(
                    "PolymlpFC: type(supercell) must be" " dict or phonopy supercell"
                )

        elif phono3py_yaml is not None:
            if self._verbose:
                print("Supercell is read from:", phono3py_yaml, flush=True)
            (self._supercell_ph, self._disps, self._st_dicts) = parse_phono3py_yaml_fcs(
                phono3py_yaml, use_phonon_dataset=use_phonon_dataset
            )
            self._supercell = phonopy_cell_to_structure(self._supercell_ph)

        else:
            raise ValueError(
                "PolymlpFC: supercell or phonon3py_yaml "
                "is required for initialization"
            )
        self._N = len(self._supercell_ph.symbols)
        return self

    def _compute_forces(self):
        _, forces, _ = self.prop.eval_multiple(self._structures)
        _, residual_forces, _ = self.prop.eval(self._supercell)
        for f in forces:
            f -= residual_forces
        return np.array(forces)

    def sample(
        self,
        n_samples: int = 100,
        displacements: float = 0.001,
        is_plusminus: bool = False,
    ):
        """Sample displacements.

        Parameters
        ----------
        n_samples: Number of supercells sampled.
        displacements: Displacement magnitude in angstroms.
        is_plusminus: Consider plus and minus displacements.

        self._disps: Displacements shape=(n_str, 3, n_atom).
        """

        self._disps, self._structures = generate_random_const_displacements(
            self._supercell,
            n_samples=n_samples,
            displacements=displacements,
            is_plusminus=is_plusminus,
        )
        return self

    def run_geometry_optimization(
        self,
        gtol: float = 1e-5,
        method: Literal["CG", "BFGS"] = "CG",
    ):
        """Run geometry optimization.

        Parameters
        ----------
        gtol: Tolerance for gradient.
        method: Optimization method.
        """

        if self._verbose:
            print("Running geometry optimization", flush=True)

        try:
            minobj = GeometryOptimization(
                self._supercell,
                relax_cell=False,
                relax_volume=False,
                relax_positions=True,
                with_sym=True,
                properties=self.prop,
                verbose=self._verbose,
            )
        except ValueError:
            print("Warning: No geomerty optimization is performed.")
            return self

        minobj.run(gtol=gtol, method=method)
        if self._verbose:
            print("Residual forces:", flush=True)
            print(minobj.residual_forces.T, flush=True)
            print("E0:", minobj.energy, flush=True)
            print("n_iter:", minobj.n_iter, flush=True)
            print("Fractional coordinate changes:", flush=True)
            diff_positions = self._supercell.positions - minobj.structure.positions
            print(diff_positions.T, flush=True)
            print("Success:", minobj.success, flush=True)

        if minobj.success:
            self._supercell = minobj.structure
            self._supercell_ph = structure_to_phonopy_cell(self._supercell)
            if self._disps is not None:
                self.displacements = self._disps

        return self

    def run_fc(
        self,
        orders: tuple = (2, 3),
        batch_size: int = 100,
        use_mkl: bool = True,
        is_compact_fc: bool = True,
    ):
        """Construct fc basis and solve FCs."""

        if self._cutoff is not None:
            cutoff = dict()
            for order in orders:
                cutoff[order] = self._cutoff
        else:
            cutoff = None

        self._symfc = Symfc(
            self._supercell_ph,
            displacements=self._disps.transpose((0, 2, 1)),
            forces=self._forces.transpose((0, 2, 1)),
            cutoff=cutoff,
            use_mkl=use_mkl,
            log_level=self._verbose,
        )
        self._symfc.run(
            orders=orders, batch_size=batch_size, is_compact_fc=is_compact_fc
        )
        for order in orders:
            if order == 2:
                self._fc2 = self._symfc.force_constants[2]
            elif order == 3:
                self._fc3 = self._symfc.force_constants[3]
            elif order == 4:
                self._fc4 = self._symfc.force_constants[4]
        return self

    def run(
        self,
        disps: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        orders: tuple = (2, 3),
        batch_size: int = 100,
        write_fc: bool = True,
        use_mkl: bool = True,
        is_compact_fc: bool = True,
    ):
        """Calculate forces using polymlp and estimate FCs."""

        if disps is not None:
            self.displacements = disps

        if forces is None:
            if self._verbose:
                print("Computing forces using polymlp", flush=True)
            t1 = time.time()
            self.forces = self._compute_forces()
            t2 = time.time()
            if self._verbose:
                print(" elapsed time (computing forces) =", t2 - t1, flush=True)
        else:
            self.forces = forces

        t1 = time.time()
        self.run_fc(
            orders=orders,
            batch_size=batch_size,
            is_compact_fc=is_compact_fc,
            use_mkl=use_mkl,
        )
        t2 = time.time()
        if self._verbose:
            print("Time (Symfc basis and solver)", t2 - t1, flush=True)
            for order in orders:
                if order == 2:
                    print(
                        "Basis size (FC2):",
                        self._symfc.basis_set[2].basis_set.shape,
                        flush=True,
                    )
                elif order == 3:
                    print(
                        "Basis size (FC3):",
                        self._symfc.basis_set[3].basis_set.shape,
                        flush=True,
                    )
                elif order == 4:
                    print(
                        "Basis size (FC4):",
                        self._symfc.basis_set[4].basis_set.shape,
                        flush=True,
                    )

        if write_fc:
            self.save_fc()

        return self

    def save_fc(self):
        if self._fc2 is not None:
            if self._verbose:
                print("writing fc2.hdf5", flush=True)
            write_fc2_to_hdf5(self._fc2)
        if self._fc3 is not None:
            if self._verbose:
                print("writing fc3.hdf5", flush=True)
            write_fc3_to_hdf5(self._fc3)

    @property
    def displacements(self) -> np.ndarray:
        return self._disps

    @property
    def forces(self) -> np.ndarray:
        return self._forces

    @property
    def structures(self) -> PolymlpStructure:
        return self._structures

    @displacements.setter
    def displacements(self, disps: np.ndarray):
        """Set displacements, shape=(n_str, 3, n_atom)."""
        if not disps.shape[1] == 3 or not disps.shape[2] == self._N:
            raise ValueError("Displacements must have a shape of " "(n_str, 3, n_atom)")
        self._disps = disps
        self._structures = get_structures_from_displacements(
            self._disps,
            self._supercell,
        )

    @forces.setter
    def forces(self, f: np.ndarray):
        """Set forces, shape=(n_str, 3, n_atom)."""
        if not f.shape[1] == 3 or not f.shape[2] == self._N:
            raise ValueError("Forces must have a shape of " "(n_str, 3, n_atom)")
        self._forces = f

    @structures.setter
    def structures(self, structures):
        self._structures = structures

    @property
    def fc2(self):
        return self._fc2

    @property
    def fc3(self):
        return self._fc3

    @property
    def fc4(self):
        return self._fc4

    @property
    def symfc_obj(self):
        return self._symfc

    @property
    def supercell_phonopy(self):
        return self._supercell_ph

    @property
    def supercell(self):
        return self._supercell

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        print("Cutoff radius:", value, "(ang.)", flush=True)
        self._cutoff = value
        self._fc_cutoff = FCCutoff(self._supercell_ph, cutoff=value)
        return self
