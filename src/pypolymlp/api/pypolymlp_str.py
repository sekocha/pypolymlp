"""API class for generating structures for DFT calculations."""

from typing import Literal, Optional, Union

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.str_gen.strgen import (
    StructureGenerator,
    set_structure_id,
    write_structures,
)


class PolymlpStructureGenerator:
    """API class for generating structures for DFT calculations."""

    def __init__(
        self,
        base_structures: Union[PolymlpStructure, list[PolymlpStructure]] = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        base_structures: Base structures for generating random structures.
        """
        self._verbose = verbose
        self._structures = None
        self._random_structures = []
        self._strgen_instances = []

        if base_structures is not None:
            self.structures = base_structures

    def load_poscars(self, poscars: Union[str, list[str]]) -> list[PolymlpStructure]:
        """Parse POSCAR files.

        Returns
        -------
        structures: list[PolymlpStructure], Structures.
        """
        if isinstance(poscars, str):
            poscars = [poscars]
        self.structures = parse_structures_from_poscars(poscars)
        return self.structures

    def load_structures_from_files(
        self,
        poscars: Optional[Union[str, list[str]]] = None,
    ):
        """Parse structure files.

        Only POSCAR files are available.

        Returns
        -------
        structures: list[PolymlpStructure], Structures.
        """
        if poscars is None:
            raise RuntimeError("Structure files not found.")

        self.structures = self.load_poscars(poscars)
        return self.structures

    def save_random_structures(self, path="./poscars"):
        """Save random structures."""
        base_info = []
        for gen in self._strgen_instances:
            base_dict = {
                "id": gen.name,
                "size": list(gen.supercell_size),
                "n_atoms": list(gen.n_atoms),
            }
            base_info.append(base_dict)

        write_structures(self._random_structures, base_info, path=path)

    def init_generator(self, max_natom: int = 150):
        """Initialize generator and construct supercells of base structures.

        Parameters
        ----------
        max_natom: Maximum number of atoms in structures.

        """
        if self._verbose:
            print("Construct supercells", flush=True)
        self._strgen_instances = []
        for st in self._structures:
            if self._verbose:
                print(st.name, flush=True)
            gen = StructureGenerator(st, natom_ub=max_natom)
            self._strgen_instances.append(gen)
            if self._verbose:
                print("-----------------------", flush=True)
                print("-", st.name, flush=True)
                gen.print_size()

        return self

    def run_standard_algorithm(
        self,
        n_samples: int = 100,
        max_distance: float = 1.5,
    ):
        """Generate random structures from base structures using a standard algorithm.

        In the standard algorithm, displacements in i-th structure are given by
            disp = max([(i + 1) / n_samples]**3 * max_distance, 0.01).

        Parameters
        ----------
        n_samples: Number of structures generated from a single POSCAR file
                   using a standard algorithm.
        max_distance: Maximum distance of displacement distributions.

        """
        for gen in self._strgen_instances:
            structures = gen.random_structure(
                n_str=n_samples,
                max_disp=max_distance,
                vol_ratio=1.0,
            )
            structures = set_structure_id(structures, gen.name, "standard")
            self._random_structures.extend(structures)

        return self

    def run_density_algorithm(
        self,
        n_samples: int = 100,
        distance: float = 0.2,
        vol_lb: float = 0.6,
        vol_ub: float = 0.9,
        vol_algorithm: Optional[Literal["low_auto", "high_auto"]] = None,
    ):
        """Generate random structures using low- and high-density algorithms.

        Parameters
        ----------
        n_samples: Number of structures generated from a single POSCAR file
                   using a standard algorithm.
        max_distance: Distance of displacement distributions.

        """
        if vol_algorithm == "low_auto":
            vol_lb, vol_ub = 1.1, 4.0
            mode = "low density"
        elif vol_algorithm == "high_auto":
            vol_lb, vol_ub = 0.6, 0.9
            mode = "high density"

        for gen in self._strgen_instances:
            structures = gen.sample_density(
                n_str=n_samples,
                disp=distance,
                vol_lb=vol_lb,
                vol_ub=vol_ub,
            )
            structures = set_structure_id(structures, gen.name, mode)
            self._random_structures.extend(structures)
        return self

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return base structures."""
        return self._structures

    @structures.setter
    def structures(
        self, structures: Union[PolymlpStructure, list[PolymlpStructure]]
    ) -> list[PolymlpStructure]:
        """Set structures."""
        if isinstance(structures, PolymlpStructure):
            self._structures = [structures]
        elif isinstance(structures, list):
            self._structures = structures
        else:
            raise RuntimeError("Invalid structure type.")

    @property
    def random_structures(self) -> list[PolymlpStructure]:
        """Return random structures."""
        return self._random_structures
