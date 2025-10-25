"""Class for computing EOS."""

from typing import Optional

import numpy as np

from pypolymlp.calculator.properties import Properties
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.structure_utils import isotropic_volume_change


class PolymlpEOS:
    """Class for computing EOS."""

    def __init__(
        self,
        unitcell: PolymlpStructure,
        pot: Optional[str] = None,
        params: Optional[PolymlpStructure] = None,
        coeffs: Optional[np.ndarray] = None,
        properties: Optional[Properties] = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        unitcell: unitcell in PolymlpStructure format
        pot: polymlp file.
        params: Parameters for polymlp.
        coeffs: Polymlp coefficients.
        properties: Properties object.

        Any one of pot, (params, coeffs), and properties is needed.
        """

        if properties is not None:
            self.prop = properties
        else:
            self.prop = Properties(pot=pot, params=params, coeffs=coeffs)

        self._verbose = verbose
        self._unitcell = unitcell

        self._eos_data = None
        self._eos_fit_data = None
        self._b0 = None
        self._e0 = None
        self._v0 = None

    def _set_eps(self, eps_min=0.7, eps_max=2.0, eps_int=0.03, fine_grid=True):

        if fine_grid == False:
            eps_seq = np.arange(eps_min, eps_max + 0.01, eps_int)
        else:
            eps_seq = []
            if eps_min < 0.9 and eps_max > 1.1:
                eps_seq.extend(list(np.arange(eps_min, 0.9, eps_int)))
                eps_seq.extend(list(np.arange(0.9, 1.1, eps_int / 3)))
                eps_seq.extend(list(np.arange(1.1, eps_max, eps_int)))
            elif eps_min < 0.9 and eps_max < 1.1:
                eps_seq.extend(list(np.arange(eps_min, 0.9, eps_int)))
                eps_seq.extend(list(np.arange(0.9, eps_max, eps_int / 3)))
            elif eps_min > 0.9 and eps_max > 1.1:
                eps_seq.extend(list(np.arange(eps_min, 1.1, eps_int / 3)))
                eps_seq.extend(list(np.arange(1.1, eps_max, eps_int)))
            else:
                eps_seq.extend(list(np.arange(eps_min, eps_max, eps_int / 3)))

        return eps_seq

    def run_eos_fit(self, volumes: np.ndarray, energies: np.ndarray):
        """Fit EOS curve."""
        return self.run_eos_fit_phonopy(volumes, energies)

    def run_eos_fit_phonopy(self, volumes: np.ndarray, energies: np.ndarray):
        """Fit EOS curve using phonopy."""

        from phonopy.qha.core import BulkModulus
        from phonopy.units import EVAngstromToGPa

        if self._verbose:
            print("EOS fitting using Vinet EOS equation")

        bm = BulkModulus(volumes=volumes, energies=energies, eos="vinet")
        self._b0 = bm.bulk_modulus * EVAngstromToGPa
        self._e0 = bm.energy
        self._v0 = bm.equilibrium_volume

        v_min, v_max = min(volumes), max(volumes)
        extrapolation = (v_max - v_min) * 0.1
        v_lb = v_min - extrapolation
        v_ub = v_max + extrapolation

        volumes_eval = np.arange(v_lb, v_ub, 0.01)
        fitted = bm._eos(volumes_eval, *bm.get_parameters())
        eos_fit_data = np.stack([volumes_eval, fitted]).T
        return eos_fit_data

    def run_eos_fit_pymatgen(self, volumes: np.ndarray, energies: np.ndarray):
        """Fit EOS curve using pymatgen."""

        from pymatgen.analysis.eos import EOS

        if self._verbose:
            print("EOS fitting using Vinet EOS equation")
        eos = EOS(eos_name="vinet")
        eos_fit = eos.fit(volumes, energies)
        self._b0 = eos_fit.b0_GPa
        self._e0 = eos_fit.e0
        self._v0 = eos_fit.v0

        v_min, v_max = min(volumes), max(volumes)
        extrapolation = (v_max - v_min) * 0.1
        v_lb = v_min - extrapolation
        v_ub = v_max + extrapolation

        volumes_eval = np.arange(v_lb, v_ub, 0.01)
        eos_fit_data = [[vol, eos_fit.func(vol)] for vol in volumes_eval]
        eos_fit_data = np.array(eos_fit_data)
        return eos_fit_data

    def run(
        self,
        eps_min=0.7,
        eps_max=2.0,
        eps_int=0.03,
        fine_grid=True,
        eos_fit=False,
    ):

        eps_list = self._set_eps(
            eps_min=eps_min,
            eps_max=eps_max,
            eps_int=eps_int,
            fine_grid=fine_grid,
        )
        structures = [
            isotropic_volume_change(self._unitcell, eps=eps) for eps in eps_list
        ]

        energies, _, _ = self.prop.eval_multiple(structures)
        volumes = np.array([st.volume for st in structures])
        if self._verbose:
            print(" eps =", np.array(eps_list))
        self._eos_data = np.array([volumes, energies]).T

        if eos_fit:
            try:
                self._eos_fit_data = self.run_eos_fit(volumes, energies)
            except:
                print("Warning: EOS fitting failed.")

        return self

    def _write_data_2d(self, data, stream, tag="volume_helmholtz"):

        print("  " + tag + ":", file=stream)
        for d in data:
            print("  -", list(d), file=stream)
        print("", file=stream)

    def write_eos_yaml(self, write_eos_fit=True, filename="polymlp_eos.yaml"):

        f = open(filename, "w")

        if self._b0 is not None:
            print("equilibrium:", file=f)
            print("  bulk_modulus:", float(self._b0), file=f)
            print("  free_energy: ", self._e0, file=f)
            print("  volume:      ", self._v0, file=f)
            print(
                "  n_atoms:     ",
                list(self._unitcell.n_atoms),
                file=f,
            )
            print("", file=f)
            print("", file=f)

        print("eos_data:", file=f)
        print("", file=f)
        self._write_data_2d(self._eos_data, f, tag="volume_helmholtz")
        print("", file=f)

        if write_eos_fit and self._eos_fit_data is not None:
            print("eos_fit_data:", file=f)
            print("", file=f)
            self._write_data_2d(self._eos_fit_data, f, tag="volume_helmholtz")
            print("", file=f)

        f.close()
