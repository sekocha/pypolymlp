"""Class for generating random structures."""

import os

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.vasp_utils import write_poscar_file


def write_structures(
    structures: list[PolymlpStructure],
    base_info: list[dict[str]],
    path: str = "poscars",
):
    """Save logs and structures to POSCAR files."""
    os.makedirs(path, exist_ok=True)
    f = open("polymlp_str_samples.yaml", "w")
    if len(base_info) > 0:
        print("prototypes:", file=f)
        for base_dict in base_info:
            print("- id:             ", base_dict["id"], file=f)
            print("  supercell_size: ", base_dict["size"], file=f)
            print("  n_atoms:        ", base_dict["n_atoms"], file=f)
        print("", file=f)

    print("structures:", file=f)
    for i, st in enumerate(structures):
        idx = str(i + 1).zfill(5)
        print("- id:", idx, file=f)
        print("  base:", st.base, file=f)
        print("  mode:", st.mode, file=f)

        filename = path + "/poscar-" + idx
        header = "pypolymlp: random-" + idx
        write_poscar_file(st, filename=filename, header=header)

    f.close()


def set_structure_id(structures: list[PolymlpStructure], poscar: str, mode: str):
    """Set additional tags to structures."""
    for st in structures:
        st.base = poscar
        st.mode = mode
    return structures


def set_volume_eps_array(
    n_samples: int = 30,
    eps_min: float = 0.8,
    eps_max: float = 2.0,
    dense_equilibrium: bool = False,
):
    """Generate a sequence of ratios used to change volume."""
    if not dense_equilibrium:
        return np.linspace(eps_min, eps_max, n_samples)

    interval_dense = 0.2 / (n_samples + 1)
    if eps_min > 0.9:
        raise RuntimeError("eps_min must be lower than 0.9.")
    if eps_max < 1.1:
        raise RuntimeError("eps_max must be higher than 1.1.")

    dense_min = 0.9 + interval_dense
    dense_max = 1.1 - interval_dense

    eps_array1 = np.linspace(eps_min, 0.9, n_samples // 3)
    eps_array2 = np.linspace(dense_min, dense_max, n_samples)
    if n_samples // 3 == 1:
        eps_array3 = [(1.1 + eps_max) / 2]
    else:
        eps_array3 = np.linspace(1.1, eps_max, n_samples // 3)
    return np.concatenate([eps_array1, eps_array2, eps_array3])


class StructureGenerator:
    """Class for generating random structures."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        natom_lb: int = 48,
        natom_ub: int = 150,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: Unitcell structure.
        natom_lb: Lower bound of number of atoms.
        natom_ub: Upper bound of number of atoms.
        """
        if natom_lb > natom_ub:
            raise ValueError("natom_lb > n_atom_ub.")

        self._unitcell = unitcell
        self._natom_lb = natom_lb
        self._natom_ub = natom_ub

        self._supercell = self._set_supercell()
        self._name = unitcell.name
        self._info = None

    def _set_supercell(self) -> PolymlpStructure:
        """Set supercell size and expand unitcell into supercell."""
        self._size = self._find_supercell_size_nearly_isotropic()
        self._supercell = supercell_diagonal(self._unitcell, self._size)
        self._supercell.axis_inv = np.linalg.inv(self._supercell.axis)
        return self._supercell

    def _find_supercell_size_nearly_isotropic(self) -> list[int]:
        """Find a diagonal supercell size enabling nealy-isotropic supercell."""
        axis = self._unitcell.axis
        total_n_atoms = sum(self._unitcell.n_atoms)

        len_axis = [np.linalg.norm(axis[:, i]) for i in range(3)]
        ratio1 = len_axis[0] / len_axis[1]
        ratio2 = len_axis[0] / len_axis[2]
        ratio = np.array([1, ratio1, ratio2])

        cand = np.arange(1, 11)
        size = [1, 1, 1]
        for c in cand:
            size_trial = np.maximum(np.round(ratio * c).astype(int), [1, 1, 1])
            n_total = total_n_atoms * np.prod(size_trial)
            if n_total >= self._natom_lb:
                if n_total <= self._natom_ub:
                    size = size_trial
                break
            size = size_trial
        return size

    def sample_random_single_structure(
        self,
        disp: float,
        vol_ratio: float = 1.0,
    ) -> PolymlpStructure:
        """Generate single random structure."""

        cell = self._supercell
        total_n_atoms = cell.positions.shape[1]
        axis_ratio = pow(vol_ratio, 1.0 / 3.0)

        axis_add = (np.random.rand(3, 3) * 2.0 - 1) * disp
        positions_add = (np.random.rand(3, total_n_atoms) * 2.0 - 1) * disp
        positions_add = cell.axis_inv @ positions_add

        str_rand = PolymlpStructure(
            axis=cell.axis * axis_ratio + axis_add,
            positions=cell.positions + positions_add,
            n_atoms=cell.n_atoms,
            elements=cell.elements,
            types=cell.types,
        )
        return str_rand

    def sample_random_structures(
        self,
        n_str: int = 10,
        max_disp: float = 1.0,
        vol_ratio: float = 1.0,
    ):
        """Generate random structures.

        disp = max([(i + 1) / n_str]**3 * max_disp, 0.01)
        """
        st_array = []
        for i in range(n_str):
            disp_function1 = 0.01
            disp_function2 = pow((i + 1) / float(n_str), 3.0) * max_disp
            disp = max(disp_function1, disp_function2)
            str_rand = self.sample_random_single_structure(disp, vol_ratio=vol_ratio)
            st_array.append(str_rand)
        return st_array

    def random_structure_algo2(
        self,
        n_str: int = 10,
        max_disp: float = 0.3,
        vol_ratio: float = 1.0,
    ):
        """Generate random structures.

        Deprecated.
        disp = [(i + 1) / n_str] * max_disp
        """
        st_array = []
        for i in range(n_str):
            disp = (i + 1) * max_disp / float(n_str)
            str_rand = self.sample_random_single_structure(disp, vol_ratio=vol_ratio)
            st_array.append(str_rand)
        return st_array

    def sample_density(
        self,
        n_str: int = 10,
        disp: float = 0.2,
        vol_lb: float = 1.1,
        vol_ub: float = 4.0,
    ):
        """Generate random structures with various densities."""
        st_array = []
        for vol_ratio in np.linspace(vol_lb, vol_ub, n_str):
            str_rand = self.sample_random_single_structure(disp, vol_ratio=vol_ratio)
            st_array.append(str_rand)
        return st_array

    def print_size(self):
        """Print supercell size."""
        print("  supercell size:      ", list(self._size), flush=True)
        print("  n_atoms (supercell): ", list(self._supercell.n_atoms), flush=True)

    @property
    def supercell(self):
        """Return supercell."""
        return self._supercell

    @property
    def supercell_size(self):
        """Return supercell size for base structure."""
        return self._size

    @property
    def name(self):
        """Return name of base structure."""
        return self._name

    @property
    def n_atoms(self):
        """Return number of atoms in supercell."""
        return self._supercell.n_atoms
