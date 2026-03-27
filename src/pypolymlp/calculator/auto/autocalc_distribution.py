"""API Class for evaluating accuracy of distributions."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.calculator.auto.autocalc_utils import AutoCalcBase
from pypolymlp.calculator.auto.figures_formation import plot_binary_formation_energies
from pypolymlp.calculator.auto.figures_properties import (
    plot_energy_distribution,
    plot_prototype_prediction,
)
from pypolymlp.calculator.auto.structures_types import get_structure_types
from pypolymlp.calculator.compute_formation_energies import (
    PolymlpFormationEnergies,
    find_endmembers,
)
from pypolymlp.calculator.properties import Properties
from pypolymlp.core.interface_vasp import parse_properties_from_vaspruns
from pypolymlp.utils.atomic_energies.atomic_energies import get_atomic_energies


@dataclass
class FormationEnergyData:
    """Dataclass for storing formation energy data."""

    names: list
    data_dft_all: np.ndarray
    data_dft_convex: np.ndarray
    data_mlp_all: np.ndarray
    data_mlp_convex: np.ndarray
    data_mlp_go_all: Optional[np.ndarray] = None
    data_mlp_go_convex: Optional[np.ndarray] = None

    def save_data(self, filename: str):
        """Save data to a yaml file."""
        convex1 = self._collect_equiv_data(self.data_dft_all, self.data_dft_convex)
        convex2 = self._collect_equiv_data(self.data_mlp_all, self.data_mlp_convex)
        if self.data_mlp_go_all is None:
            convex3 = None
        else:
            convex3 = self._collect_equiv_data(
                self.data_mlp_go_all,
                self.data_mlp_go_convex,
            )

        with open(filename, "w") as f:
            print("convex_hull_dft:", file=f)
            for d in convex1:
                print("- composition:", list(d[:-2].astype(float)), file=f)
                print("  delta:      ", float(d[-2]), file=f)
                print("  prototype:  ", d[-1], file=f)
            print(file=f)

            print("convex_hull_mlp:", file=f)
            for d in convex2:
                print("- composition:", list(d[:-2].astype(float)), file=f)
                print("  delta:      ", float(d[-2]), file=f)
                print("  prototype:  ", d[-1], file=f)
            print(file=f)

            if convex3 is not None:
                print("convex_hull_mlp_geometry_optimization:", file=f)
                for d in convex3:
                    print("- composition:", list(d[:-2].astype(float)), file=f)
                    print("  delta:      ", float(d[-2]), file=f)
                    print("  prototype:  ", d[-1], file=f)
                print(file=f)

            print("formation_energies_dft:", file=f)
            for d, n in zip(self.data_dft_all, self.names):
                print("- prototype:", n, file=f)
                print("  data:     ", list(d), file=f)
            print(file=f)

            print("formation_energies_mlp:", file=f)
            for d, n in zip(self.data_mlp_all, self.names):
                print("- prototype:", n, file=f)
                print("  data:     ", list(d), file=f)
            print(file=f)

            if self.data_mlp_go_all is not None:
                print("formation_energies_mlp_geometry_optimization:", file=f)
                for d, n in zip(self.data_mlp_go_all, self.names):
                    print("- prototype:", n, file=f)
                    print("  data:     ", list(d), file=f)
                print(file=f)
        return self

    def _collect_equiv_data(
        self,
        data_all: np.ndarray,
        data_convex: np.ndarray,
        decimals: int = 4,
    ):
        """Collect all equivalent data on convex hull.."""

        tol = 10 ** (-decimals)
        convex = []
        for de1 in data_convex:
            ids = np.where(np.all(np.isclose(data_all, de1, atol=tol), axis=1))[0]
            for i in ids:
                data = list(np.round(data_all[i], decimals + 2))
                data.append(self.names[i])
                convex.append(data)
        convex = np.array(convex)

        order = [-1, -2] + list(range(0, convex.shape[1] - 2))
        keys = [convex[:, i] for i in order]
        convex = convex[np.lexsort(keys)]
        return convex


@dataclass
class EnergyData:
    """Dataclass for storing energy and structure data."""

    vaspruns: Optional[list] = None
    structures: Optional[list] = None
    n_atom: Optional[np.ndarray] = None
    names: Optional[list] = None
    energies_mlp: Optional[np.ndarray] = None
    energies_dft: Optional[np.ndarray] = None
    energies_mlp_per_atom: Optional[np.ndarray] = None
    energies_dft_per_atom: Optional[np.ndarray] = None

    def get_comparison_data(self, decimals: int = 6):
        """Return data for comparison between DFT and MLP."""
        if self.energies_dft_per_atom is None or self.energies_mlp_per_atom is None:
            raise RuntimeError("Energies per atom not found.")
        if self.names is None:
            raise RuntimeError("Names not found.")

        data = np.stack(
            [
                np.round(self.energies_dft_per_atom, decimals),
                np.round(self.energies_mlp_per_atom, decimals),
                self.names,
            ]
        ).T
        data = data[data[:, 0].argsort()]
        return data


class AutoCalcDistribution(AutoCalcBase):
    """API Class for evaluating accuracy of distributions."""

    def __init__(
        self,
        properties: Properties,
        path_output: str = ".",
        functional: str = "PBE",
        verbose: bool = False,
    ):
        """Init method.

        Parameters
        ----------
        properties: Properties instance.
        """
        super().__init__(properties, path_output=path_output, verbose=verbose)
        self._prop = properties
        self._verbose = verbose

        self._comparison = None
        self._distribution_train = None
        self._distribution_test = None
        self._distribution_formation_train = None
        self._distribution_formation_test = None
        self._formation_energies = None

        self._formation_mlp = None
        self._formation_dft = None
        if self._n_types > 1:
            self._formation_mlp = PolymlpFormationEnergies(properties=self._prop)
            self._formation_dft = PolymlpFormationEnergies(
                elements=self._element_strings
            )
        self._functional = functional

    def _parse_vaspruns(self, vaspruns: list):
        """Parse vasprun files."""
        if len(vaspruns) == 0:
            raise RuntimeError("Empty vasprun data.")
        structures, (energies, _, _) = parse_properties_from_vaspruns(vaspruns)
        energies = self._apply_atomic_energies(structures, energies)
        n_atom = np.array([np.sum(st.n_atoms) for st in structures])
        return structures, energies, n_atom

    def _apply_atomic_energies(self, structures: list, energies: np.ndarray):
        """Calculate atomic energies for structures."""
        atom_e = get_atomic_energies(
            elements=self._element_strings,
            functional=self._functional,
            return_dict=True,
        )
        atomic_energies = np.array(
            [np.sum([atom_e[ele] for ele in st.elements]) for st in structures]
        )
        energies -= atomic_energies
        return energies

    def _eval_energies(self, vaspruns: list):
        """Evaluate MLP energies and DFT cohesive energies."""
        structures, energies_dft, n_atom = self._parse_vaspruns(vaspruns)
        self._calc.structures = structures
        energies_mlp, _, _ = self._calc.eval()
        energy_data = EnergyData(
            vaspruns=vaspruns,
            structures=structures,
            n_atom=n_atom,
            energies_dft=energies_dft,
            energies_mlp=energies_mlp,
            energies_dft_per_atom=energies_dft / n_atom,
            energies_mlp_per_atom=energies_mlp / n_atom,
        )
        return energy_data

    def _set_structure_names(
        self,
        vaspruns: Optional[list] = None,
        icsd_ids: Optional[list] = None,
    ):
        """Set structure names."""
        if icsd_ids is None:
            return np.array(vaspruns)

        structure_types = get_structure_types()
        names = [
            "ICSD-" + str(i) + "-[" + structure_types[str(i)] + "]" for i in icsd_ids
        ]
        return np.array(names)

    def compare_with_dft(
        self,
        vaspruns: list,
        icsd_ids: Optional[list] = None,
        filename_suffix: Optional[str] = None,
    ):
        """Calculate properties for DFT structures."""
        if self._verbose:
            print("Compute energies for structures.", flush=True)

        energy_data = self._eval_energies(vaspruns)
        energy_data.names = self._set_structure_names(vaspruns, icsd_ids=icsd_ids)
        self._comparison = energy_data.get_comparison_data()

        filename = self._path_header + "comparison"
        if filename_suffix is not None:
            filename += "_" + filename_suffix
        filename += ".dat"
        header = "DFT (eV/atom), MLP (eV/atom), ID"
        np.savetxt(filename, self._comparison, fmt="%s", header=header)
        return self._comparison

    def plot_comparison_with_dft(
        self,
        system: str,
        pot_id: str,
        filename_suffix: Optional[str] = None,
    ):
        """Plot comparison of mlp predictions with dft."""
        if self._comparison is None:
            raise RuntimeError("Comparison data not found.")

        plot_prototype_prediction(
            self._comparison,
            system,
            pot_id,
            path_output=self._path_output,
            filename_suffix=filename_suffix,
        )
        return self

    def _set_end_members(self, vaspruns: list):
        """Set endmembers."""
        energy_data = self._eval_energies(vaspruns)
        ends = find_endmembers(
            energy_data.structures,
            energy_data.energies_dft,
            self._element_strings,
        )
        end_energies = np.array([e for _, e in ends])
        self._formation_dft.define_end_members(energies=end_energies)
        if self._verbose:
            print("Energies of end members (DFT):", end_energies, flush=True)

        ends = find_endmembers(
            energy_data.structures,
            energy_data.energies_mlp,
            self._element_strings,
        )
        end_energies = np.array([e for _, e in ends])
        self._formation_mlp.define_end_members(energies=end_energies)
        if self._verbose:
            print("Energies of end members (MLP):", end_energies, flush=True)
        return energy_data

    def calc_formation_energies(
        self,
        vaspruns,
        icsd_ids: Optional[list] = None,
        geometry_optimization: bool = False,
    ):
        """Run formation energy calculations."""
        if self._n_types == 1:
            return self
        if self._verbose:
            print("Compute formation energies.", flush=True)

        energy_data = self._set_end_members(vaspruns)
        names = self._set_structure_names(vaspruns, icsd_ids=icsd_ids)

        delta_e_dft = self._formation_dft.compute(
            energy_data.structures,
            energies=energy_data.energies_dft,
        )
        delta_e_dft_convex = self._formation_dft.convex_hull()

        delta_e_mlp = self._formation_mlp.compute(
            energy_data.structures,
            energies=energy_data.energies_mlp,
        )
        delta_e_mlp_convex = self._formation_mlp.convex_hull()

        delta_e_mlp_go, delta_e_mlp_go_convex = None, None
        if geometry_optimization:
            res = self._run_formation_energy_with_optimization(energy_data)
            delta_e_mlp_go, delta_e_mlp_go_convex = res

        self._formation_energies = FormationEnergyData(
            names,
            delta_e_dft,
            delta_e_dft_convex,
            delta_e_mlp,
            delta_e_mlp_convex,
            delta_e_mlp_go,
            delta_e_mlp_go_convex,
        )
        filename = self._path_header + "formation_energy.yaml"
        self._formation_energies.save_data(filename=filename)
        return self

    def _run_formation_energy_with_optimization(self, energy_data: EnergyData):
        """Plot comparison of mlp predictions with dft."""
        structures_success = []
        energies_success = []
        for st in energy_data.structures:
            self._calc.structures = st
            self._calc.init_geometry_optimization(relax_cell=True, relax_volume=True)
            try:
                energy, _, _ = self._calc.run_geometry_optimization(method="CG")
                structures_success.append(self._calc.converged_structure)
                energies_success.append(energy)
            except:
                pass

        delta_e_mlp_go = self._formation_mlp.compute(
            structures=structures_success,
            energies=energies_success,
        )
        delta_e_mlp_go_convex = self._formation_mlp.convex_hull()
        return (delta_e_mlp_go, delta_e_mlp_go_convex)

    def plot_binary_formation_energies(self, system: str, pot_id: str):
        """Plot formation energies."""
        if self._n_types != 2:
            raise RuntimeError("System is not binary.")
        if self._formation_energies is None:
            raise RuntimeError("Formation energy data not found.")

        plot_binary_formation_energies(
            system,
            pot_id,
            self._formation_energies.data_dft_all,
            self._formation_energies.data_dft_convex,
            self._formation_energies.data_mlp_all,
            self._formation_energies.data_mlp_convex,
            self._formation_energies.data_mlp_go_all,
            self._formation_energies.data_mlp_go_convex,
            path_output=self._path_output,
        )
        return self

    def calc_energy_distribution(
        self,
        vaspruns_train: list,
        vaspruns_test: list,
    ):
        """Calculate properties for structures in training and test datasets."""
        if self._verbose:
            print("Compute energies for training and test data.", flush=True)

        energy_data_train = self._eval_energies(vaspruns_train)
        energy_data_test = self._eval_energies(vaspruns_test)

        self._distribution_train = self._stack_data(
            energy_data_train.energies_dft_per_atom,
            energy_data_train.energies_mlp_per_atom,
        )
        self._distribution_test = self._stack_data(
            energy_data_test.energies_dft_per_atom,
            energy_data_test.energies_mlp_per_atom,
        )

        size = self._distribution_train.shape[0] + self._distribution_test.shape[0]
        with open(self._path_header + "size.dat", "w") as f:
            print(size, file=f)

        self._calc_formation_energy_distribution(energy_data_train, energy_data_test)
        return self

    def _calc_formation_energy_distribution(
        self,
        energy_data_train: EnergyData,
        energy_data_test: EnergyData,
    ):
        if self._n_types == 1:
            return self
        if not self._formation_dft.has_end_members:
            return self
        if not self._formation_mlp.has_end_members:
            return self

        delta_e_train_dft = self._formation_dft.compute(
            energy_data_train.structures,
            energies=energy_data_train.energies_dft,
        )
        delta_e_train_mlp = self._formation_mlp.compute(
            energy_data_train.structures,
            energies=energy_data_train.energies_mlp,
        )
        delta_e_test_dft = self._formation_dft.compute(
            energy_data_test.structures,
            energies=energy_data_test.energies_dft,
        )
        delta_e_test_mlp = self._formation_mlp.compute(
            energy_data_test.structures,
            energies=energy_data_test.energies_mlp,
        )
        self._distribution_formation_train = self._stack_data(
            delta_e_train_dft[:, -1],
            delta_e_train_mlp[:, -1],
        )
        self._distribution_formation_test = self._stack_data(
            delta_e_test_dft[:, -1],
            delta_e_test_mlp[:, -1],
        )
        return self

    def plot_energy_distribution(self, system: str, pot_id: str):
        """Plot comparison of mlp predictions with dft."""
        if self._distribution_train is None or self._distribution_test is None:
            raise RuntimeError("Distribution data not found.")

        plot_energy_distribution(
            self._distribution_train,
            self._distribution_test,
            system,
            pot_id,
            path_output=self._path_output,
        )

        if self._distribution_formation_train is None:
            return self
        if self._distribution_formation_test is None:
            return self

        plot_energy_distribution(
            self._distribution_formation_train,
            self._distribution_formation_test,
            system,
            pot_id,
            path_output=self._path_output,
            filename_suffix="formation",
            header="Formation energy distribution",
        )
        return self

    def _stack_data(self, data1: np.ndarray, data2: np.ndarray):
        """Stack two dataset."""
        stacked = np.stack([data1, data2]).T
        return stacked
