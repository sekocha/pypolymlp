# Systematic Property Calculations

## Property Calculations for Prototype Structures

> **Note:** This feature is available only for elemental and binary alloy systems.
> Requires version 0.19.1 or later.

Several properties of prototype structures prepared in `pypolymlp` can be automatically calculated.
For elemental systems, properties are calculated for 18 prototype structures, while for binary alloy systems, calculations are performed for 69 structures, including elemental end-member structures.

The calculated properties include:

- Equilibrium structure
- Potential energy of the equilibrium structure
- Equilibrium volume
- Bulk modulus
- Equation of state (EOS)
- Lattice constants
- Elastic constants
- Phonon density of states
- Thermodynamic properties from QHA calculations

When using the command-line interface, simply run `pypolymlp-autocalc` by specifying the potential file name, as follows:

```shell
pypolymlp-autocalc --pot polymlp.yaml
```

When using the Python API, the following example may be helpful:

```python
from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc

pot = "polymlp.yaml"
calc = PypolymlpAutoCalc(pot=pot, verbose=True)
calc.run_prototypes()
calc.save_prototypes()
```

## Comparison with DFT Calculations

### Elemental System
```python
import glob
from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc

pot = "polymlp.yaml"
vaspruns = glob.glob("DFT-prototypes/Ti/*/vasprun.xml.polymlp")
icsd_ids = [v.split("/")[-2] for v in vaspruns]

calc = PypolymlpAutoCalc(pot=pot, verbose=True)
calc.run_prototypes()
calc.set_prototypes_from_DFT(vaspruns, icsd_ids)
calc.save_prototypes()

calc.calc_comparison_with_dft(
    vaspruns=vaspruns,
    icsd_ids=icsd_ids,
)
calc.plot_comparison_with_dft("Ti", "polymlp-00001")

vaspruns_train = glob.glob("DFT-train/Ti/*/vasprun.xml.polymlp")
vaspruns_test = glob.glob("DFT-test/Ti/*/vasprun.xml.polymlp")
calc.calc_energy_distribution(vaspruns_train, vaspruns_test)
calc.plot_energy_distribution("Ti", "polymlp-00001")
```

### Binary Alloy System
```python
import numpy as np
import glob

from pypolymlp.api.pypolymlp_autocalc import PypolymlpAutoCalc

pot = "polymlp.yaml"

vaspruns1 = glob.glob("DFT-prototype/Ag-Au/*/vasprun.xml.polymlp")
icsd_ids1 = [v.split("/")[-2] for v in vaspruns1]
vaspruns2 = glob.glob("DFT-prototype/Ag/*/vasprun.xml.polymlp")
icsd_ids2 = [v.split("/")[-2] for v in vaspruns2]
vaspruns3 = glob.glob("DFT-prototype/Au/*/vasprun.xml.polymlp")
icsd_ids3 = [v.split("/")[-2] for v in vaspruns3]

vaspruns_all= []
vaspruns_all.extend(vaspruns1)
vaspruns_all.extend(vaspruns2)
vaspruns_all.extend(vaspruns3)
icsd_ids_all = []
icsd_ids_all.extend(icsd_ids1)
icsd_ids_all.extend(icsd_ids2)
icsd_ids_all.extend(icsd_ids3)

calc = PypolymlpAutoCalc(pot=pot, verbose=True)
calc.run_prototypes()
calc.set_prototypes_from_DFT(vaspruns1, icsd_ids1)
calc.save_prototypes()

calc.calc_comparison_with_dft(
    vaspruns=vaspruns1,
    icsd_ids=icsd_ids1,
    filename_suffix="Ag-Au",
)
calc.plot_comparison_with_dft(
    "Ag-Au",
    "polymlp-00001",
    filename_suffix="Ag-Au",
)
calc.calc_comparison_with_dft(
    vaspruns=vaspruns2,
    icsd_ids=icsd_ids2,
    filename_suffix="Ag",
)
calc.plot_comparison_with_dft(
    "Ag in Ag-Au",
    "polymlp-00001",
    filename_suffix="Ag",
)
calc.calc_comparison_with_dft(
    vaspruns=vaspruns3,
    icsd_ids=icsd_ids3,
    filename_suffix="Au",
)
calc.plot_comparison_with_dft(
    "Au in Ag-Au",
    "polymlp-00001",
    filename_suffix="Au",
)

calc.calc_formation_energies(
    vaspruns=vaspruns_all,
    icsd_ids=icsd_ids_all,
    geometry_optimization=False,
)
calc.plot_binary_formation_energies("Ag-Au", "polymlp-00001")

vaspruns_train = glob.glob("DFT-train/Ag-Au/*/vasprun.xml.polymlp")
vaspruns_test = glob.glob("DFT-test/Ag-Au/*/vasprun.xml.polymlp")
calc.calc_energy_distribution(vaspruns_train, vaspruns_test)
calc.plot_energy_distribution("Ag-Au", "polymlp-00001")
```
