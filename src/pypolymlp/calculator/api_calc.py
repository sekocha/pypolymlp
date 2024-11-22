"""API Class for calculating properties."""

from typing import Union, Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.calculator.properties import Properties
from pypolymlp.calculator.compute_elastic import PolymlpElastic


class PolymlpCalc:
    """API Class for calculating properties."""

    def __init__(
        self,
        pot: Union[str, list[str]] = None,
        params: Union[PolymlpParams, list[PolymlpParams]] = None,
        coeffs: Union[np.ndarray, list[np.ndarray]] = None,
        properties: Optional[Properties] = None,
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """
        if pot is None and params is None and properties is None:
            raise RuntimeError("Poly. MLP not defined.")
        if properties is None:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)
        else:
            self._prop = properties
 
        self._verbose = verbose

    def eval(self, st: PolymlpStructure):
        """Evaluate properties for a single structure.

        Returns
        -------
        e: Energy. shape=(n_str,), unit: eV/supercell
        f: Forces. shape=(n_str, 3, natom), unit: eV/angstrom.
        s: Stress tensors. shape=(n_str, 6), 
            unit: eV/supercell in the order of xx, yy, zz, xy, yz, zx.
        """
        # from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure
        # st = phonopy_cell_to_structure(str_ph)
        if isinstance(st, PolymlpStructure):
            return self._prop.eval(st)
        elif isinstance(st, list):
            return self._prop.eval_multiple(structures)
        raise RuntimeError("Invalid structure type.")

    def save_properties(self):
        """Save properties.

        Numpy files of polymlp_energies.npy, polymlp_forces.npy, 
        and polymlp_stress_tensors.npy are generated.
        They contain the energy values, forces, and stress tensors 
        for structures used for the latest run of self.eval.
        """
        self._prop.save(verbose=self._verbose)
        return self

    def print_single_properties(self):
        """Print properties for a single structure."""
        self._prop.print_single()
        return self

    def run_features(
        self, 
        structures: list[PolymlpStructure], 
        develop_infile: Optional[str] = None,
    ):
        """Compute features.

        Parameters
        ----------
        structures: Structures for computing features.
        develop_infile: A pypolymlp input file for developing MLP.
        """
        if develop_infile is None:

    def run_elastic_constants(
        self, structure: PolymlpStructure, poscar: str,
    ):
        """Run elastic constant calculations.

        pymatgen is required.

        Returns
        -------
        elastic_constants: Elastic constants in GPa. shape=(6,6).
        """
        self.unicell = structure
        self._elastic = PolymlpElastic(
            unitcell = structure,
            unitcell = poscar,
            properties=self._prop, 
            verbose=self._verbose,
        )
        self._elastic.run()
        return self._elastic.elastic_constants

    def write_elastic_constants(self, filename="polymlp_elastic.yaml"):
        """Save elastic constants to a file."""
        self._elastic.write_elastic_constants(filename = filename)

    @property
    def properties(self) -> Properties:
        """Return Properties object."""
        return self._prop

    @property
    def params(self) -> PolymlpParams:
        """Return parameters."""
        return self._prop.params

    @property
    def energies(self) -> np.ndarray:
        """Return energies from the final calculation."""
        return self._prop.energies

    @property
    def forces(self) -> list:
        """Return forces from the final calculation."""
        return self._prop.forces

    @property
    def stresses(self) -> np.ndarray:
        """Return stress tensors from the final calculation."""
        return self._prop.stresses

    @property
    def stresses_gpa(self) -> np.ndarray:
        """Return stress tensors in GPa from the final calculation."""
        return self._prop.stresses_gpa

    @property
    def structures(self) -> list[PolymlpStructure]:
        """Return structures for the final calculation."""
        return self._prop._structures

    @property
    def unitcell(self) -> PolymlpStructure:
        """Return unit cell."""
        return self._unitcell

    @setter.unitcell
    def unitcell(self, cell):
        """Set unit cell."""
        self._unitcell = cell

    @property
    def elastic_constants(self) -> np.ndarray:
        """Return elastic constants."""
        return self._elastic.elastic_constants

