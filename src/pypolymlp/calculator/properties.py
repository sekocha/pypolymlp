"""Class for calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.io_polymlp import load_mlp
from pypolymlp.cxx.lib import libmlpcpp


def find_active_atoms(
    structures: list[PolymlpStructure],
    element_order: list[str],
):
    """Reconstruct structures only using active atoms."""
    structures_active = []
    active_atoms_all = []
    active_bools = []
    for st in structures:
        active_atoms = np.array(
            [i for i, ele in enumerate(st.elements) if ele in element_order]
        )
        types = np.array([element_order.index(st.elements[i]) for i in active_atoms])
        n_atoms = [np.count_nonzero(types == i) for i in range(len(element_order))]

        if len(active_atoms) > 0:
            st_active = PolymlpStructure(
                axis=st.axis,
                positions=st.positions[:, active_atoms],
                n_atoms=n_atoms,
                elements=np.array(st.elements)[active_atoms],
                types=types,
            )
            structures_active.append(st_active)
            active_atoms_all.append(active_atoms)
            active_bools.append(True)
        else:
            active_bools.append(False)

    return structures_active, active_atoms_all, np.array(active_bools)


def convert_stresses_in_gpa(stresses: np.ndarray, structures: list[PolymlpStructure]):
    """Calculate stress tensor values in GPa."""
    volumes = np.array([st.volume for st in structures])
    stresses_gpa = np.zeros(stresses.shape)
    for i in range(6):
        stresses_gpa[:, i] = stresses[:, i] / volumes * 160.21766208
    return stresses_gpa


class PropertiesSingle:
    """Class for calculating properties using a single polymlp model."""

    def __init__(
        self, pot: str = None, params: PolymlpParams = None, coeffs: np.ndarray = None
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        """

        if pot is not None:
            self._params, self._coeffs = load_mlp(filename=pot)
        else:
            self._params = params
            self._coeffs = coeffs

        self._params.element_swap = False
        self.obj = libmlpcpp.PotentialPropertiesFast(
            self._params.as_dict(), self._coeffs
        )

    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure.

        Return
        ------
        energy: unit: eV/supercell
        force: unit: eV/angstrom (3, n_atom)
        stress: unit: eV/supercell: (6) in the order of xx, yy, zz, xy, yz, zx
        """
        if self._params.type_full:
            st_calc = update_types([st], self._params.element_order)[0]
        else:
            st_calc, active_atoms, _ = find_active_atoms(
                [st], self._params.element_order
            )
            if len(st_calc) > 0:
                st_calc = st_calc[0]
                active_atoms = active_atoms[0]
            else:
                st_calc = None

        if st_calc is not None:
            positions_c = st_calc.axis @ st_calc.positions
            self.obj.eval(st_calc.axis, positions_c, st_calc.types, use_openmp)

            energy = self.obj.get_e()
            force = np.array(self.obj.get_f()).T
            stress = np.array(self.obj.get_s())

            if self._params.type_full == False:
                force_full = np.zeros((3, len(st.types)))
                force_full[:, active_atoms] = force
                force = force_full
        else:
            energy = 0.0
            force = np.zeros((3, len(st.types)))
            stress = np.zeros(6)

        return energy, force, stress

    def eval_multiple(self, structures: list[PolymlpStructure], verbose: bool = False):
        """Evaluate properties for multiple structures.

        Return
        ------
        energies: unit: eV/supercell (n_str)
        forces: unit: eV/angstrom (n_str, 3, n_atom)
        stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                    unit: eV/supercell
        """
        if verbose:
            print(
                "Properties calculations for",
                len(structures),
                "structures: Using a fast algorithm",
            )
        if self._params.type_full:
            structures_calc = update_types(structures, self._params.element_order)
        else:
            structures_calc, active_atoms, active_bools = find_active_atoms(
                structures, self._params.element_order
            )

        if len(structures_calc) > 0:
            axis_array = [st.axis for st in structures_calc]
            types_array = [st.types for st in structures_calc]
            positions_c_array = [st.axis @ st.positions for st in structures_calc]

            # PotentialProperties.eval_multiple: Return
            # ------------------------------------------
            # energies = obj.get_e(), (n_str)
            # forces = obj.get_f(), (n_str, n_atom, 3)
            # stresses = obj.get_s(), (n_str, 6)
            #             in the order of xx, yy, zz, xy, yz, zx
            self.obj.eval_multiple(axis_array, positions_c_array, types_array)

            energies = np.array(self.obj.get_e_array())
            stresses = np.array(self.obj.get_s_array())
            forces = [np.array(f).T for f in self.obj.get_f_array()]
        else:
            energies, forces, stresses = [], [], []

        if self._params.type_full == False:
            energies_full, forces_full, stresses_full = [], [], []
            i = 0
            for iall, active in enumerate(active_bools):
                st = structures[iall]
                f_full = np.zeros((3, len(st.types)))
                if active:
                    atoms = active_atoms[i]
                    f_full[:, atoms] = forces[i]
                    energies_full.append(energies[i])
                    forces_full.append(f_full)
                    stresses_full.append(stresses[i])
                    i += 1
                else:
                    energies_full.append(0.0)
                    forces_full.append(f_full)
                    stresses_full.append(np.zeros(6))

            energies = np.array(energies_full)
            forces = forces_full
            stresses = np.array(stresses_full)

        return energies, forces, stresses

    @property
    def params(self):
        return self._params


class PropertiesHybrid:
    """Class for calculating properties using a hybrid polymlp model."""

    def __init__(
        self,
        pot: str = None,
        params: list[PolymlpParams] = None,
        coeffs: list[np.ndarray] = None,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        """

        if pot is not None:
            if not isinstance(pot, list):
                raise ValueError("Parameters in PropertiesHybrid must be lists.")
            self.props = [PropertiesSingle(pot=p) for p in pot]
        else:
            if not isinstance(params, list) or not isinstance(coeffs, list):
                raise ValueError("Parameters in PropertiesHybrid must be lists.")
            self.props = [
                PropertiesSingle(params=p, coeffs=c) for p, c in zip(params, coeffs)
            ]

    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure."""
        energy, force, stress = self.props[0].eval(st, use_openmp=use_openmp)
        for prop in self.props[1:]:
            e_single, f_single, s_single = prop.eval(st, use_openmp=use_openmp)
            energy += e_single
            force += f_single
            stress += s_single
        return energy, force, stress

    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        energies, forces, stresses = self.props[0].eval_multiple(structures)
        for prop in self.props[1:]:
            e_single, f_single, s_single = prop.eval_multiple(structures)
            energies += e_single
            for i, f1 in enumerate(f_single):
                forces[i] += f1
            stresses += s_single
        return energies, forces, stresses

    @property
    def params(self):
        return [prop.params for prop in self.props]


class Properties:
    """Class for calculating properties."""

    def __init__(
        self,
        pot: str = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.

        Any one of pot and (params, coeffs) is needed.
        """

        if pot is not None:
            if isinstance(pot, list):
                if len(pot) > 1:
                    self.prop = PropertiesHybrid(pot=pot)
                else:
                    self.prop = PropertiesSingle(pot=pot[0])
            else:
                self.prop = PropertiesSingle(pot=pot)
        else:
            if isinstance(params, list) and isinstance(coeffs, list):
                if len(params) > 1 and len(coeffs) > 1:
                    self.prop = PropertiesHybrid(params=params, coeffs=coeffs)
                else:
                    self.prop = PropertiesSingle(
                        params_dict=params[0], coeffs=coeffs[0]
                    )
            else:
                self.prop = PropertiesSingle(params=params, coeffs=coeffs)

    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure."""
        e, f, s = self.prop.eval(st, use_openmp=use_openmp)
        self._e, self._f, self._s = [e], [f], [s]
        self._structures = [st]
        return e, f, s

    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        self._e, self._f, self._s = self.prop.eval_multiple(structures)
        self._structures = structures
        return self._e, self._f, self._s

    def eval_phonopy(self, str_ph, use_openmp: bool = True):
        """Evaluate properties for a single structure in phonopy format."""
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

        st = phonopy_cell_to_structure(str_ph)
        e, f, s = self.prop.eval(st, use_openmp=use_openmp)
        self._e, self._f, self._s = [e], [f], [s]
        self._structures = [st]
        return e, f, s

    def eval_multiple_phonopy(self, str_ph_list):
        """Evaluate properties for multiple structures in phonopy format."""
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

        structures = [phonopy_cell_to_structure(str_ph) for str_ph in str_ph_list]
        self._e, self._f, self._s = self.prop.eval_multiple(structures)
        self._structures = structures
        return self._e, self._f, self._s

    def save(self, verbose=False):
        np.save("polymlp_energies.npy", self.energies)
        np.save("polymlp_forces.npy", self.forces)
        np.save("polymlp_stress_tensors.npy", self.stresses_gpa)
        if len(self.forces) == 1:
            np.savetxt("polymlp_energies.dat", self.energies, fmt="%f")

        if verbose:
            print(
                "polymlp_energies.npy, polymlp_forces.npy,",
                "and polymlp_stress_tensors.npy are generated.",
            )
        return self

    def print_single(self):
        np.set_printoptions(suppress=True)
        print("Energy:", self.energies[0], "(eV/cell)")
        print("Forces:")
        for i, f in enumerate(self.forces[0].T):
            print("- atom", i, ":", f)
        stress = self.stresses_gpa[0]
        print("Stress tensors:")
        print("- xx, yy, zz:", stress[0:3])
        print("- xy, yz, zx:", stress[3:6])
        print("---------")
        return self

    @property
    def params(self):
        return self.prop.params

    @property
    def energies(self):
        return self._e

    @property
    def forces(self):
        return self._f

    @property
    def stresses(self):
        return self._s

    @property
    def stresses_gpa(self):
        return convert_stresses_in_gpa(self._s, self._structures)


def set_instance_properties(
    pot: Union[str, list[str]] = None,
    params: Union[PolymlpParams, list[PolymlpParams]] = None,
    coeffs: Union[np.ndarray, list[np.ndarray]] = None,
    properties: Optional[Properties] = None,
    require_mlp: bool = True,
):
    """Set instance of Properties class.

    Parameters
    ----------
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties instance.

    Any one of pot, (params, coeffs), and properties is needed.
    """
    if require_mlp:
        if pot is None and params is None and properties is None:
            raise RuntimeError("polymlp not defined.")

    if properties is not None:
        return properties
    elif pot is not None or params is not None:
        return Properties(pot=pot, params=params, coeffs=coeffs)
    return None
