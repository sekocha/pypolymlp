"""Class for calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.properties_hybrid import PropertiesHybrid
from pypolymlp.calculator.properties_single import PropertiesSingle
from pypolymlp.calculator.utils.properties_base import PropertiesBase
from pypolymlp.calculator.utils.properties_utils import convert_stresses_in_gpa
from pypolymlp.core.data_format import PolymlpParamsSingle, PolymlpStructure
from pypolymlp.core.params import PolymlpParams


class Properties(PropertiesBase):
    """Class for calculating properties."""

    def __init__(
        self,
        pot: Optional[Union[str, list]] = None,
        params: Optional[PolymlpParams] = None,
        coeffs: Optional[list[np.ndarray]] = None,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.

        Any one of pot and (params, coeffs) is needed.
        """

        self._pot = None
        if pot is not None:
            self._pot = pot
            if isinstance(pot, (list, tuple, np.ndarray)):
                if len(pot) > 1:
                    self._prop = PropertiesHybrid(pot=pot)
                else:
                    self._prop = PropertiesSingle(pot=pot[0])
            else:
                self._prop = PropertiesSingle(pot=pot)
        else:
            self._pot = "Coefficients"
            if len(params) == 1:
                if len(coeffs) != 1:
                    self._prop = PropertiesSingle(params=params.params, coeffs=coeffs)
                else:
                    self._prop = PropertiesSingle(
                        params=params.params, coeffs=coeffs[0]
                    )
            else:
                if len(params) != len(coeffs):
                    raise RuntimeError("Length of params and coeffs not consistent.")
                self._prop = PropertiesHybrid(params=params, coeffs=coeffs)

    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure."""
        e, f, s = self._prop.eval(st, use_openmp=use_openmp)
        self._e, self._f, self._s = [e], [f], [s]
        self._structures = [st]
        return e, f, s

    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        self._e, self._f, self._s = self._prop.eval_multiple(structures)
        self._structures = structures
        return self._e, self._f, self._s

    def eval_phonopy(self, str_ph, use_openmp: bool = True):
        """Evaluate properties for a single structure in phonopy format."""
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

        st = phonopy_cell_to_structure(str_ph)
        e, f, s = self._prop.eval(st, use_openmp=use_openmp)
        self._e, self._f, self._s = [e], [f], [s]
        self._structures = [st]
        return e, f, s

    def eval_multiple_phonopy(self, str_ph_list):
        """Evaluate properties for multiple structures in phonopy format."""
        from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

        structures = [phonopy_cell_to_structure(str_ph) for str_ph in str_ph_list]
        self._e, self._f, self._s = self._prop.eval_multiple(structures)
        self._structures = structures
        return self._e, self._f, self._s

    def save(self, verbose: bool = False):
        """Save properties to files."""
        np.save("polymlp_energies.npy", self.energies)
        np.save("polymlp_stress_tensors.npy", self.stresses_gpa)
        try:
            np.save("polymlp_forces.npy", self.forces)
        except:
            for i, force in enumerate(self.forces):
                np.save("polymlp_forces_" + str(i + 1).zfill(5) + ".npy", force)

        if len(self.forces) == 1:
            np.savetxt("polymlp_energies.dat", self.energies, fmt="%f")

        if verbose:
            print(
                "polymlp_energies.npy, polymlp_forces*.npy,",
                "and polymlp_stress_tensors.npy are generated.",
                flush=True,
            )
        return self

    def print_single(self):
        """Print properties for single structure calculation."""
        np.set_printoptions(suppress=True)
        print("Energy:", self.energies[0], "(eV/cell)", flush=True)
        print("Forces (eV/ang):", flush=True)
        for i, f in enumerate(self.forces[0].T):
            print("- atom", i, ":", f, flush=True)

        stress = self.stresses[0]
        print("Stress tensors (eV/cell):", flush=True)
        print("- xx, yy, zz:", stress[0:3], flush=True)
        print("- xy, yz, zx:", stress[3:6], flush=True)
        stress = self.stresses_gpa[0]
        print("Stress tensors (GPa):", flush=True)
        print("- xx, yy, zz:", stress[0:3], flush=True)
        print("- xy, yz, zx:", stress[3:6], flush=True)
        print("---------", flush=True)
        return self

    @property
    def elements(self):
        """Return elements."""
        return self._prop.elements

    @property
    def params(self):
        """Return parameters."""
        if isinstance(self._prop.params, PolymlpParamsSingle):
            return PolymlpParams(self._prop.params)
        return self._prop.params

    @property
    def pot(self):
        """Return potential path."""
        return self._pot

    @property
    def energies(self):
        """Return energies."""
        return self._e

    @property
    def forces(self):
        """Return forces."""
        return self._f

    @property
    def stresses(self):
        """Return stresses in eV/cell."""
        return self._s

    @property
    def stresses_gpa(self):
        """Return stresses in GPa."""
        return convert_stresses_in_gpa(self._s, self._structures)


def initialize_polymlp_calculator(
    pot: Optional[Union[str, list[str]]] = None,
    params: Optional[PolymlpParams] = None,
    coeffs: Optional[list[np.ndarray]] = None,
    properties: Optional[Properties] = None,
    return_none: bool = False,
):
    """Initialize calculator of polymlp.

    Parameters
    ----------
    pot: polymlp file.
    params: Parameters for polymlp.
    coeffs: Polymlp coefficients.
    properties: Properties object.

    Any one of pot, (params, coeffs), and properties is needed.
    """
    if all(x is None for x in (properties, pot, params, coeffs)):
        if return_none:
            return None
        raise RuntimeError("Polymlp not provided.")
    elif properties is not None:
        return properties

    if params is not None:
        if coeffs is None:
            raise RuntimeError("Coefficients not provided.")
        if len(params) > 1 and len(params) != len(coeffs):
            raise RuntimeError("Length of params and coeffs not consistent.")
    return Properties(pot=pot, params=params, coeffs=coeffs)
