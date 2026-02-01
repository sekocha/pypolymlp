"""Class for calculating FCs using polymlp."""

import time
from typing import Optional, Union

import numpy as np
from phono3py.file_IO import write_fc2_to_hdf5, write_fc3_to_hdf5
from symfc import Symfc
from symfc.utils.cutoff_tools import FCCutoff

from pypolymlp.calculator.properties import Properties
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

        self._prop = None
        if properties is not None:
            self._prop = properties
        else:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)
        self._verbose = verbose

        self._initialize_supercell(
            supercell=supercell,
            phono3py_yaml=phono3py_yaml,
            use_phonon_dataset=use_phonon_dataset,
        )
        if cutoff is not None:
            # TODO: Implement order dependence of cutoff
            self.cutoff = cutoff
        else:
            self._cutoff = None
            self._fc_cutoff = None

        self._fc2 = None
        self._fc3 = None
        self._fc4 = None
        self._symfc = None
        self._disps = None
        self._forces = None

    def _initialize_supercell(
        self,
        supercell: Optional[PolymlpStructure] = None,
        phono3py_yaml: Optional[str] = None,
        use_phonon_dataset: bool = False,
    ):
        """Initialize supercell."""
        if supercell is None and phono3py_yaml is None:
            raise RuntimeError("Supercell and phonon3py_yaml not found.")

        if supercell is not None:
            self._supercell = supercell
            self._supercell_ph = structure_to_phonopy_cell(supercell)
        elif phono3py_yaml is not None:
            if self._verbose:
                print("Supercell is read from:", phono3py_yaml, flush=True)
            res = parse_phono3py_yaml_fcs(
                phono3py_yaml, use_phonon_dataset=use_phonon_dataset
            )
            self._supercell_ph, self._disps, self._structures = res
            self._supercell = phonopy_cell_to_structure(self._supercell_ph)

        self._N = len(self._supercell_ph.symbols)
        return self

    def _compute_forces(self):
        """Compute forces and subtract residul forces."""
        if self._structures is None:
            raise RuntimeError("Structures not found.")

        _, forces, _ = self._prop.eval_multiple(self._structures)
        _, residual_forces, _ = self._prop.eval(self._supercell)
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

    def run_fc(
        self,
        orders: tuple = (2, 3),
        batch_size: int = 100,
        use_mkl: bool = True,
        is_compact_fc: bool = True,
    ):
        """Construct fc basis and solve FCs."""
        if self._disps is None:
            RuntimeError("Displacements not found.")

        if self._forces is None:
            RuntimeError("Forces not found.")

        cutoff = None
        if self._cutoff is not None:
            cutoff = dict()
            for order in orders:
                cutoff[order] = self._cutoff

        self._symfc = Symfc(
            self._supercell_ph,
            displacements=self._disps.transpose((0, 2, 1)),
            forces=self._forces.transpose((0, 2, 1)),
            cutoff=cutoff,
            use_mkl=use_mkl,
            log_level=self._verbose,
        )
        self._symfc.run(
            orders=orders,
            batch_size=batch_size,
            is_compact_fc=is_compact_fc,
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
                shape = self._symfc.basis_set[order].blocked_basis_set.shape
                if order == 2:
                    print("Basis size (FC2):", shape, flush=True)
                elif order == 3:
                    shape = self._symfc.basis_set[3].blocked_basis_set.shape
                    print("Basis size (FC3):", shape, flush=True)
                elif order == 4:
                    print("Basis size (FC4):", shape, flush=True)

        if write_fc:
            self.save_fc()

        return self

    def save_fc(self):
        """Save force constants."""
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
        """Return displacements."""
        return self._disps

    @property
    def forces(self) -> np.ndarray:
        """Return forces."""
        return self._forces

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return supercell structures with displacements."""
        return self._structures

    @displacements.setter
    def displacements(self, disps: np.ndarray):
        """Set displacements, shape=(n_str, 3, n_atom)."""
        if not disps.shape[1] == 3 or not disps.shape[2] == self._N:
            raise RuntimeError("Displacements not (n_str, 3, n_atom) shape.")

        self._disps = disps
        self._structures = get_structures_from_displacements(
            self._disps,
            self._supercell,
        )

    @forces.setter
    def forces(self, f: np.ndarray):
        """Set forces, shape=(n_str, 3, n_atom)."""
        if not f.shape[1] == 3 or not f.shape[2] == self._N:
            raise RuntimeError("Forces not (n_str, 3, n_atom) shape.")

        self._forces = f

    @structures.setter
    def structures(self, structures):
        """Set supercell structures with displacements."""
        self._structures = structures

    @property
    def fc2(self):
        """Return 2nd-order FCs."""
        return self._fc2

    @property
    def fc3(self):
        """Return 3rd-order FCs."""
        return self._fc3

    @property
    def fc4(self):
        """Return 4th-order FCs."""
        return self._fc4

    @property
    def symfc_obj(self):
        """Return symfc instance."""
        return self._symfc

    @property
    def supercell_phonopy(self):
        """Return supercell in phonopy format."""
        return self._supercell_ph

    @property
    def supercell(self):
        """Return supercell in pypolymlp format."""
        return self._supercell

    @property
    def cutoff(self):
        """Return cutoff distance."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        """Set cutoff distance."""
        if self._verbose:
            print("Cutoff radius:", value, "(ang.)", flush=True)
        self._cutoff = value
        self._fc_cutoff = FCCutoff(self._supercell_ph, cutoff=value)
