"""API class for generating structures for systematic SSCHA calculations."""

import os
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.strgen_sym import StructureGeneratorSym
from pypolymlp.utils.structure_utils import (
    multiple_isotropic_volume_changes,
    multiple_random_deformation,
)
from pypolymlp.utils.vasp_utils import write_poscar_file


class PypolymlpSSCHAStructureGenerator:
    """API class for generating structures for systematic SSCHA calculations."""

    def __init__(
        self,
        structure: Optional[PolymlpStructure] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        base_structure: Base structures for generating random structures.
        """
        self._verbose = verbose
        if structure is not None:
            self._structure = structure
        self._strgen = None
        self._structure_samples = None

    def load_poscar(self, poscar: str) -> PolymlpStructure:
        """Parse POSCAR file.

        Returns
        -------
        structure: PolymlpStructure for poscar file.
        """
        self._structure = Poscar(poscar).structure
        return self._structure

    def sample_volumes(
        self,
        n_samples: int = 30,
        eps_min: float = 0.8,
        eps_max: float = 1.5,
        fix_axis: bool = True,
        max_deform: float = 0.1,
    ):
        """Generate structures with isotropic volume changes.

        Parameters
        ----------
        n_samples: Number of structures generated from a single structure.
        eps_min: Minimum ratio of volume.
        eps_max: Maximum ratio of volume.
        max_deform: Maximum magnitude of lattice deformation.
        """
        if self._verbose:
            print("Volume ratios:", flush=True)
            print(np.linspace(eps_min, eps_max, num=n_samples), flush=True)

        self._structure_samples = multiple_isotropic_volume_changes(
            self._structure,
            eps_min=eps_min,
            eps_max=eps_max,
            n_eps=n_samples,
        )
        if not fix_axis:
            self._structure_samples = multiple_random_deformation(
                self._structure_samples,
                max_deform=max_deform,
            )

        return self

    def sample_sym(
        self,
        n_samples: int = 100,
        fix_axis: bool = False,
        fix_positions: bool = False,
        max_deform: float = 0.1,
        max_distance: float = 0.1,
    ):
        """Generate random structures with symmetry constraints.

        Parameters
        ----------
        n_samples: Number of sampled structures.
        max_deform: Maximum magnitude of lattice deformation.
        max_distance: Maximum magnitude of atomic displacements.
        """
        self._strgen = StructureGeneratorSym(
            self._structure,
            fix_axis=fix_axis,
            fix_positions=fix_positions,
            verbose=self._verbose,
        )
        self._strgen.run(
            n_samples=n_samples,
            max_deform=max_deform,
            max_distance=max_distance,
        )
        self._structure_samples = self._strgen.structures
        return self

    def save_structures(self, path="./poscars"):
        """Save structures."""
        os.makedirs(path, exist_ok=True)
        for i, st in enumerate(self._structure_samples):
            write_poscar_file(st, filename=path + "/POSCAR-" + str(i + 1).zfill(4))
        return self

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return random structures."""
        return self._structure_samples

    @property
    def basis_sets(self) -> list[PolymlpStructure]:
        """Return basis sets for axis and positions."""
        return self._strgen.basis_sets
