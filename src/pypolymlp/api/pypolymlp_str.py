"""API class for generating structures for DFT calculations."""

import copy
from typing import Literal, Optional, Union

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.displacements import generate_random_const_displacements
from pypolymlp.core.interface_vasp import parse_structures_from_poscars
from pypolymlp.core.strgen import (
    StructureGenerator,
    set_structure_id,
    set_volume_eps_array,
    write_structures,
)
from pypolymlp.utils.structure_utils import multiple_isotropic_volume_changes, supercell


class PypolymlpStructureGenerator:
    """API class for generating structures for DFT calculations."""

    def __init__(
        self,
        base_structures: Union[PolymlpStructure, list[PolymlpStructure]] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        base_structures: Base structures for generating random structures.
        """
        self._verbose = verbose
        self._structures = None
        self._supercells = None
        self._sample_structures = []
        self._strgen_instances = []

        if base_structures is not None:
            self.structures = base_structures

        if self._verbose:
            np.set_printoptions(legacy="1.21")

    def load_poscars(self, poscars: Union[str, list[str]]) -> list[PolymlpStructure]:
        """Parse POSCAR files.

        Returns
        -------
        structures: list[PolymlpStructure], Structures.
        """
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

        self.load_poscars(poscars)
        return self

    def save_random_structures(self, path: str = "./poscars"):
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
        base_structures: Union[PolymlpStructure, list[PolymlpStructure]] = None,
        supercell_size: np.ndarray = (2, 2, 2),
        use_phonopy: bool = False,
    ):
        """Initialize by constructing a supercell of a single base structure."""
        if base_structures is not None:
            self.structures = base_structures

        self._supercells = []
        for unitcell in self._structures:
            sup = supercell(
                unitcell,
                supercell_matrix=supercell_size,
                use_phonopy=use_phonopy,
            )
            self._supercells.append(sup)
        return self

    def run_const_displacements(self, n_samples: int = 100, distance: float = 0.03):
        """Generate random structures with constant magnitude of displacements.

        Parameters
        ----------
        n_samples: Number of structures generated for each supercell structure.
        distance: Magnitude of atomic displacements.

        """
        if self._supercells is None:
            raise RuntimeError("Set supercells at first.")

        for st in self._supercells:
            _, structures = generate_random_const_displacements(
                st,
                n_samples=n_samples,
                displacements=distance,
            )
            structures = set_structure_id(structures, st.name, "Displacement")
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
        if self._supercells is None:
            raise RuntimeError("Set supercells at first.")

        distances = np.linspace(distance_lb, distance_ub, num=n_samples)
        if self._verbose:
            print("Distances:", flush=True)
            print(distances, flush=True)

        if n_volumes == 1:
            for dis in distances:
                self.run_const_displacements(n_samples=1, distance=dis)
            return self

        eps_array = np.linspace(eps_min, eps_max, num=n_volumes)
        if self._verbose:
            print("Volume ratios:", flush=True)
            print(eps_array, flush=True)

        supercells_copied = copy.deepcopy(self._supercells)
        for st in supercells_copied:
            self._supercells = multiple_isotropic_volume_changes(
                st, eps_array=eps_array
            )
            for dis in distances:
                self.run_const_displacements(n_samples=1, distance=dis)
        self._supercell = supercells_copied

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
        if self._supercells is None:
            raise RuntimeError("Set supercells at first.")

        eps_array = set_volume_eps_array(
            n_samples=n_samples,
            eps_min=eps_min,
            eps_max=eps_max,
            dense_equilibrium=dense_equilibrium,
        )
        if self._verbose:
            print("Volume ratios:", flush=True)
            print(eps_array, flush=True)

        for st in self._supercells:
            structures = multiple_isotropic_volume_changes(st, eps_array=eps_array)
            structures = set_structure_id(structures, st.name, "Volume")
            self._sample_structures.extend(structures)
        return self

    def build_supercells_auto(self, max_natom: int = 150):
        """Initialize generators and construct supercells of base structures.

        Parameters
        ----------
        max_natom: Maximum number of atoms in structures.

        """
        self._strgen_instances = []
        for st in self._structures:
            gen = StructureGenerator(st, natom_ub=max_natom)
            self._strgen_instances.append(gen)
            if self._verbose:
                print("-----------------------", flush=True)
                print("-", st.name, flush=True)
                gen.print_size()

        return self

    def run_standard_algorithm(self, n_samples: int = 100, max_distance: float = 1.5):
        """Generate random structures from base structures using a standard algorithm.

        In the standard algorithm, displacements in i-th structure are given by
            disp = max([(i + 1) / n_samples]**3 * max_distance, 0.01).

        Parameters
        ----------
        n_samples: Number of structures generated from a single POSCAR file
                   using a standard algorithm.
        max_distance: Maximum distance of displacement distributions.

        """
        if len(self._strgen_instances) == 0:
            raise RuntimeError(
                "Structure generator not found. Use build_supercells_auto."
            )

        for gen in self._strgen_instances:
            structures = gen.sample_random_structures(
                n_str=n_samples,
                max_disp=max_distance,
                vol_ratio=1.0,
            )
            structures = set_structure_id(structures, gen.name, "Standard")
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
        if len(self._strgen_instances) == 0:
            raise RuntimeError(
                "Structure generator not found. Use build_supercells_auto."
            )

        if vol_algorithm == "low_auto":
            vol_lb, vol_ub = 1.1, 4.0
            mode = "Low density"
        elif vol_algorithm == "high_auto":
            vol_lb, vol_ub = 0.6, 0.9
            mode = "High density"

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
