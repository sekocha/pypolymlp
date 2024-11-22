"""API Class for calculating properties."""

from typing import Optional, Union

import numpy as np

from pypolymlp.calculator.compute_elastic import PolymlpElastic
from pypolymlp.calculator.compute_eos import PolymlpEOS
from pypolymlp.calculator.compute_features import (
    compute_from_infile,
    compute_from_polymlp_lammps,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpParams, PolymlpStructure
from pypolymlp.core.interface_vasp import parse_structures_from_poscars


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
            raise RuntimeError("polymlp not defined.")

        if properties is None:
            self._prop = Properties(pot=pot, params=params, coeffs=coeffs)
        else:
            self._prop = properties

        self._verbose = verbose
        self._unitcell = None
        self._structures = None

        self._elastic = None
        self._eos = None

    def parse_poscars(self, poscars: list[str]) -> list[PolymlpStructure]:
        """Parse POSCAR files.

        Retruns
        -------
        structures: list[PolymlpStructure], Structures.
        """
        if isinstance(poscars, str):
            poscars = [poscars]
        self.structures = parse_structures_from_poscars(poscars)
        return self.structures

    def eval(
        self,
        structures: Optional[Union[PolymlpStructure, list[PolymlpStructure]]] = None,
    ):
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
        if structures is not None:
            self.structures = structures
        return self._prop.eval_multiple(self.structures)

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
        structures: Optional[PolymlpStructure, list[PolymlpStructure]] = None,
        develop_infile: Optional[str] = None,
        features_force: bool = False,
        features_stress: bool = False,
    ):
        """Compute features.

        Parameters
        ----------
        structures: Structures for computing features.
        develop_infile: A pypolymlp input file for developing MLP.

        Return
        ------
        features: Structural features. shape=(n_str, n_features)
            if features_force == False and features_stress == False.
        """
        if structures is not None:
            self.structures = structures

        if develop_infile is None:
            features = compute_from_polymlp_lammps(
                self.structures,
                params=self.params,
                force=features_force,
                stress=features_stress,
                return_mlp_dict=False,
            )
        else:
            features = compute_from_infile(
                develop_infile,
                self.structures,
                force=features_force,
                stress=features_stress,
            )

        return features

    def run_elastic_constants(
        self,
        structure: PolymlpStructure,
        poscar: str,
    ):
        """Run elastic constant calculations.

        pymatgen is required.

        Returns
        -------
        elastic_constants: Elastic constants in GPa. shape=(6,6).
        """
        self.unicell = structure
        self._elastic = PolymlpElastic(
            unitcell=structure,
            unitcell_poscar=poscar,
            properties=self._prop,
            verbose=self._verbose,
        )
        self._elastic.run()
        return self._elastic.elastic_constants

    def write_elastic_constants(self, filename="polymlp_elastic.yaml"):
        """Save elastic constants to a file."""
        self._elastic.write_elastic_constants(filename=filename)

    def run_eos(
        self,
        structure: Optional[PolymlpStructure] = None,
        eps_min: float = 0.7,
        eps_max: float = 2.0,
        eps_step: float = 0.03,
        fine_grid: bool = True,
        eos_fit: bool = False,
    ):
        """Run EOS calculations.

        pymatgen is required if eos_fit = True.

        Parameters
        ----------
        structure: Equilibrium structure.
        eps_min: Lower bound of volume change.
        eps_max: Upper bound of volume change.
        eps_step: Interval of volume change.
        fine_grid: Use a fine grid around equilibrium structure.
        eos_fit: Fit vinet EOS curve using volume-energy data.

        volumes = np.arange(eps_min, eps_max, eps_step) * eq_volume

        Returns
        -------
        self: PolymlpCalc
        """
        if structure is not None:
            self.structures = structure

        self.unitcell = self.first_structure
        self._eos = PolymlpEOS(
            unitcell=self.unitcell,
            properties=self._prop,
            verbose=self._verbose,
        )
        self._eos.run(
            eps_min=eps_min,
            eps_max=eps_max,
            eps_int=eps_step,
            fine_grid=fine_grid,
            eos_fit=eos_fit,
        )
        return self

    def write_eos(self, filename="polymlp_eos.yaml"):
        """Save EOS to a file."""
        self._eos.write_eos_yaml(filename=filename)

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
    def unitcell(self) -> PolymlpStructure:
        """Return unit cell."""
        return self._unitcell

    @unitcell.setter
    def unitcell(self, cell):
        """Set unit cell."""
        self._unitcell = cell

    @property
    def elastic_constants(self) -> np.ndarray:
        """Return elastic constants."""
        return self._elastic.elastic_constants

    @property
    def eos_fit_data(self):
        """Return EOS fit parameters.

        Returns
        -------
        equilibrium energy, equilibrium volume, bulk modulus
        """
        return (self._eos._e0, self._eos._v0, self._eos._b0)
