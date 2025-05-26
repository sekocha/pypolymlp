"""API class for generating structures for DFT calculations."""

from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.displacements import generate_random_const_displacements
from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.strgen import StructureGenerator, set_structure_id, write_structures
from pypolymlp.utils.structure_utils import (
    multiple_isotropic_volume_changes,
    supercell_diagonal,
)


class PypolymlpStructureGenerator:
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
        self._supercell = None
        self._sample_structures = []
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

        write_structures(self._sample_structures, base_info, path=path)

    def build_supercell(
        self,
        base_structure: Optional[PolymlpStructure] = None,
        supercell_size: np.ndarray = (2, 2, 2),
        use_phonopy: bool = False,
    ):
        """Initialize by constructing a supercell of a single base structure."""
        if base_structure is not None:
            self.structures = base_structure
        unitcell = self.first_structure

        self._supercell = supercell_diagonal(
            unitcell,
            size=supercell_size,
            use_phonopy=use_phonopy,
        )
        return self

    def run_const_displacements(
        self,
        n_samples: int = 100,
        distance: float = 0.03,
    ):
        """Generate random structures with constant magnitude of displacements.

        Parameters
        ----------
        n_samples: Number of structures generated from a single structure.
        distance: Magnitude of atomic displacements.

        """
        _, structures = generate_random_const_displacements(
            self._supercell,
            n_samples=n_samples,
            displacements=distance,
        )
        structures = set_structure_id(structures, "single-str", "disp")
        self._sample_structures.extend(structures)
        return self

    def run_sequential_displacements(
        self,
        n_samples: int = 100,
        distance_lb: float = 0.01,
        distance_ub: float = 1.5,
        n_volumes: int = 1,
        eps_min: float = 0.8,
        eps_max: float = 1.3,
    ):
        """Generate random structures with constant magnitude of displacements.

        Parameters
        ----------
        n_samples: Number of structures generated from a single structure.
        distance_lb: Minimum magnitude of atomic displacements.
        distance_ub: Maximum magnitude of atomic displacements.
        n_volumes: Number of volumes.
        eps_min: Minimum ratio of volume.
        eps_max: Maximum ratio of volume.

        If n_volumes > 1, n_samples structures are generated for each volume.
        The total number of structures is n_volumes * n_samples.
        """
        distances = np.linspace(distance_lb, distance_ub, num=n_samples)
        if self._verbose:
            print("Distances:", flush=True)
            print(distances, flush=True)

        if n_volumes == 1:
            for dis in distances:
                self.run_const_displacements(n_samples=1, distance=dis)
        else:
            if self._verbose:
                print("Volume ratios:", flush=True)
                print(np.linspace(eps_min, eps_max, num=n_volumes), flush=True)

            supercells = multiple_isotropic_volume_changes(
                self._supercell,
                eps_min=eps_min,
                eps_max=eps_max,
                n_eps=n_volumes,
            )
            supercell_init = self._supercell
            for sup in supercells:
                self._supercell = sup
                for dis in distances:
                    self.run_const_displacements(n_samples=1, distance=dis)
            self._supercell = supercell_init

        return self

    def run_isotropic_volume_changes(
        self,
        n_samples: int = 30,
        eps_min: float = 0.8,
        eps_max: float = 2.0,
        dense_equilibrium: bool = False,
    ):
        """Generate structures with isotropic volume changes.

        Parameters
        ----------
        n_samples: Number of structures generated from a single structure.
        eps_min: Minimum ratio of volume.
        eps_max: Maximum ratio of volume.

        """

        if self._supercell is None:
            self._supercell = self.first_structure

        if not dense_equilibrium:
            if self._verbose:
                print("Volume ratios:", flush=True)
                print(np.linspace(eps_min, eps_max, num=n_samples), flush=True)
            structures = multiple_isotropic_volume_changes(
                self._supercell,
                eps_min=eps_min,
                eps_max=eps_max,
                n_eps=n_samples,
            )
        else:
            interval_dense = 0.2 / (n_samples + 1)
            if eps_min > 0.9:
                raise RuntimeError("eps_min must be lower than 0.9.")
            if eps_max < 1.1:
                raise RuntimeError("eps_max must be higher than 1.1.")

            dense_min = 0.9 + interval_dense
            dense_max = 1.1 - interval_dense
            if self._verbose:
                print("Volume ratios:", flush=True)
                print(np.linspace(eps_min, 0.9, num=n_samples // 3), flush=True)
                print(np.linspace(dense_min, dense_max, num=n_samples), flush=True)
                print(np.linspace(1.1, eps_max, num=n_samples // 3), flush=True)

            structures = multiple_isotropic_volume_changes(
                self._supercell,
                eps_min=eps_min,
                eps_max=0.9,
                n_eps=n_samples // 3,
            )
            structures_add = multiple_isotropic_volume_changes(
                self._supercell,
                eps_min=dense_min,
                eps_max=dense_max,
                n_eps=n_samples,
            )
            structures.extend(structures_add)
            structures_add = multiple_isotropic_volume_changes(
                self._supercell,
                eps_min=1.1,
                eps_max=eps_max,
                n_eps=n_samples // 3,
            )
            structures.extend(structures_add)

        structures = set_structure_id(structures, "single-str", "volume")
        self._sample_structures.extend(structures)
        return self

    def build_supercells_auto(self, max_natom: int = 150):
        """Initialize generators and construct supercells of base structures.

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
            self._sample_structures.extend(structures)

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
            self._sample_structures.extend(structures)
        return self

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return base structures."""
        return self._structures

    @property
    def first_structure(self) -> PolymlpStructure:
        """Return the first structure for the final calculation."""
        return self._structures[0]

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
    def sample_structures(self) -> list[PolymlpStructure]:
        """Return sample structures."""
        return self._sample_structures

    @property
    def n_samples(self):
        """Return number of sample structures."""
        return len(self._sample_structures)
