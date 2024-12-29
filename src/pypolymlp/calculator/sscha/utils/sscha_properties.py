"""Class for calculating thermodynamic properties from SSCHA results."""

from collections import defaultdict

import numpy as np
from phonopy.qha.core import BulkModulus
from phonopy.units import EVAngstromToGPa

from pypolymlp.calculator.sscha.sscha_utils import Restart


class SSCHAProperties:
    """Class for calculating thermodynamic properties from SSCHA results."""

    def __init__(self):
        """Init method."""
        self._free_energies = None
        self._entropies = None
        self._harmonic_heat_capacities = None
        self._equilibrium_volumes = None
        self._equilibrium_free_energies = None
        self._bulk_moduli = None

    def load_sscha_yamls(self, filenames: list[str]):
        """Load sscha_results.yaml files."""
        self._free_energies = defaultdict(list)
        self._entropies = defaultdict(list)
        self._harmonic_heat_capacities = defaultdict(list)
        for yamlfile in filenames:
            res = Restart(yamlfile)
            res.unit = "eV/cell"
            temp = np.round(res.temperature, decimals=2)
            self._free_energies[temp].append(
                [res.volume, res.free_energy + res.static_potential]
            )
            res.unit = "kJ/mol"
            self._entropies[temp].append([res.volume, res.entropy])
            self._harmonic_heat_capacities[temp].append(
                [res.volume, res.harmonic_heat_capacity]
            )

    def load_electron_yamls(self):
        """Add electronic free energy contribution."""
        if self._free_energies is None:
            raise RuntimeError("Call load_sscha_yamls in advance.")
        pass

    def fit_eos(self):
        """Fit EOS curves."""
        self._equilibrium_volumes = dict()
        self._equilibrium_free_energies = dict()
        self._bulk_moduli = dict()

        for temp, values in self._free_energies.items():
            values = np.array(values)
            bm = BulkModulus(
                volumes=values[:, 0],
                energies=values[:, 1],
                pressure=None,
                eos="vinet",
            )
            self._equilibrium_volumes[temp] = bm.equilibrium_volume
            self._equilibrium_free_energies[temp] = bm.energy
            self._bulk_moduli[temp] = bm.bulk_modulus * EVAngstromToGPa  # in GPa
            print(self._eos_function(bm, values[:, 0]))

    def fit_entropy(self):
        """Fit volume-entropy curves."""
        for temp, values in self._entropies.items():
            values = np.array(values)
            print(temp, values)

    def _eos_function(self, bm: BulkModulus, volumes: np.ndarray):
        """EOS function.

        Return
        ------
        Energies.
        """
        parameters = bm.get_parameters()
        energies = bm._eos(volumes, *parameters)
        return energies

    @property
    def bulk_moduli(self):
        return self._bulk_moduli


#    @property
#    def equilibrium_free_energy(self):
#        return self.__free_energy

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        nargs="*",
        type=str,
        default=None,
        help="sscha_results.yaml files",
    )
    args = parser.parse_args()

    sscha = SSCHAProperties()
    sscha.load_sscha_yamls(args.yaml)
    sscha.fit_eos()
    sscha.fit_entropy()
    print(sscha.bulk_moduli)
