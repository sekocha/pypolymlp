"""Class for calculating properties using single polymlp."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.compute_features import update_types
from pypolymlp.calculator.utils.properties_base import PropertiesBase
from pypolymlp.calculator.utils.properties_utils import find_active_atoms
from pypolymlp.core.data_format import PolymlpParamsSingle, PolymlpStructure
from pypolymlp.core.io_polymlp import load_mlp
from pypolymlp.cxx.lib import libmlpcpp


class PropertiesSingle(PropertiesBase):
    """Class for calculating properties using a single polymlp model."""

    def __init__(
        self,
        pot: Optional[str] = None,
        params: Optional[PolymlpParamsSingle] = None,
        coeffs: Optional[np.ndarray] = None,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        """

        super().__init__()
        if pot is not None:
            self._params, self._coeffs = load_mlp(filename=pot)
        else:
            self._params = params
            self._coeffs = coeffs

        params_dict = self._params.as_dict()
        self._obj = libmlpcpp.PotentialPropertiesFast(params_dict, self._coeffs)

    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure.

        Return
        ------
        energy: unit: eV/supercell
        force: unit: eV/angstrom (3, n_atom)
        stress: unit: eV/supercell: (6) in the order of xx, yy, zz, xy, yz, zx
        """
        if self._params.type_full or self._params.type_full is None:
            st_calc = update_types(st, self._params.elements)
        else:
            st_calc, active_atoms, _ = find_active_atoms([st], self._params.elements)
            if len(st_calc) > 0:
                st_calc = st_calc[0]
                active_atoms = active_atoms[0]
            else:
                st_calc = None

        if st_calc is None:
            return 0.0, np.zeros((3, len(st.types))), np.zeros(6)

        positions_c = st_calc.axis @ st_calc.positions
        self._obj.eval(st_calc.axis, positions_c, st_calc.types, use_openmp)

        energy = self._obj.get_e()
        force = np.array(self._obj.get_f()).T
        stress = np.array(self._obj.get_s())

        if self._params.type_full or self._params.type_full is None:
            return energy, force, stress

        force_full = np.zeros((3, len(st.types)))
        force_full[:, active_atoms] = force
        force = force_full

        return energy, force, stress

    def eval_multiple(self, structures: list[PolymlpStructure], verbose: bool = False):
        """Evaluate properties for multiple structures.

        Return
        ------
        energies: unit: eV/supercell (n_str)
        forces: unit: eV/angstrom (n_str, 3, n_atom)
        stresses: (n_str, 6) in the order of xx, yy, zz, xy, yz, zx
                    unit: eV/supercell

        C++ library returns energies, forces, and stresses in the following form.
        energies: shape=(n_str)
        forces = shape=(n_str, n_atom, 3)
        stresses = shape=(n_str, 6) in the order of xx, yy, zz, xy, yz, zx
        """
        if verbose:
            print(
                "Properties calculations for",
                len(structures),
                "structures: Using a fast algorithm",
                flush=True,
            )
        if self._params.type_full or self._params.type_full is None:
            structures_calc = update_types(structures, self._params.elements)
        else:
            structures_calc, active_atoms, active_bools = find_active_atoms(
                structures, self._params.elements
            )

        if len(structures_calc) == 0:
            n_str = len(structures)
            return (
                np.zeros(n_str),
                [np.zeros((3, len(st.types))) for st in structures],
                np.zeros((n_str, 6)),
            )

        axis_array = [st.axis for st in structures_calc]
        types_array = [st.types for st in structures_calc]
        positions_c_array = [st.axis @ st.positions for st in structures_calc]

        self._obj.eval_multiple(axis_array, positions_c_array, types_array)
        energies = np.array(self._obj.get_e_array())
        stresses = np.array(self._obj.get_s_array())
        forces = [np.array(f).T for f in self._obj.get_f_array()]

        if self._params.type_full or self._params.type_full is None:
            return energies, forces, stresses

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
    def elements(self):
        """Return elements."""
        return self._params.elements

    @property
    def params(self):
        """Return parameters of polymlp."""
        return self._params
