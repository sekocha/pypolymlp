"""API class for generating structures for systematic SSCHA calculations."""

from typing import Optional

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.strgen_sym import StructureGeneratorSym


class PolymlpSSCHAStructureGenerator:
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

    def load_poscar(self, poscar: str) -> PolymlpStructure:
        """Parse POSCAR file.

        Returns
        -------
        structure: PolymlpStructure for poscar file.
        """
        self._structure = Poscar(poscar).structure
        return self._structure

    def run(
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

    def save_random_structures(self, path="./poscars"):
        """Save random structures."""
        self._strgen.save_structures(path=path)

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return random structures."""
        return self._strgen.structures
