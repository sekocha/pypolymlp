# Entry Construction in the Polynomial MLP Repository

> **Note:** This feature is available only for elemental and binary alloy systems.
> Requires version 0.19.1 or later.

`pypolymlp` supports the automatic generation of entries in the [Polynomial Machine Learning Potential Repository](https://cms.mtl.kyoto-u.ac.jp/seko/mlp-repository/index.html) from systematically developed MLPs.
Before generating repository entries, a set of polynomial MLPs must be prepared.
The following components are required to create a repository entry:

- Polynomial MLPs developed with diverse parameter sets
- Results from DFT calculations for prototype structures (currently supported only for VASP)
- Results from DFT calculations for training structures (currently supported only for VASP)
- Results from DFT calculations for test structures (currently supported only for VASP)

Using the Python API, a repository entry can be generated systematically as follows.

## Elemental System
```python
import numpy as np
import glob

from pypolymlp.api.pypolymlp_repository import PypolymlpRepository

mlp_paths = [
    "grid-Ti/mlp1",
    "grid-Ti/mlp2",
    "grid-Ti/mlp3",
    "grid-Ti/mlp4",
    "grid-Ti/mlp5",
]
vaspruns = glob.glob("DFT/Ti/*/vasprun.xml.polymlp")
icsd_ids = [v.split("/")[-2] for v in vaspruns]

vaspruns_train = glob.glob("DFT-train/Ti/*/vasprun.xml.polymlp")
vaspruns_test = glob.glob("DFT-test/Ti/*/vasprun.xml.polymlp")

rep = PypolymlpRepository(mlp_paths=mlp_paths, verbose=True)

# If computational costs are not evaluated, the following must be activated.
# rep.calc_costs()

rep.extract_convex_polymlps(key="dataset/vasprun-*.xml.polymlp")
rep.calc_properties_elements(
    vaspruns_prototypes=vaspruns,
    icsd_ids=icsd_ids,
    vaspruns_train=vaspruns_train,
    vaspruns_test=vaspruns_test,
)
rep.generate_web_contents()

```

### Binary Alloy System
```python
import numpy as np
import glob

from pypolymlp.api.pypolymlp_repository import PypolymlpRepository

mlp_paths = [
    "grid-Ag-Au/mlp1",
    "grid-Ag-Au/mlp2",
    "grid-Ag-Au/mlp3",
    "grid-Ag-Au/mlp4",
    "grid-Ag-Au/mlp5",
]
vaspruns1 = glob.glob("DFT/Ag-Au/*/vasprun.xml.polymlp")
vaspruns2 = glob.glob("DFT/Ag/*/vasprun.xml.polymlp")
vaspruns3 = glob.glob("DFT/Au/*/vasprun.xml.polymlp")
icsd_ids1 = [v.split("/")[-2] for v in vaspruns1]
icsd_ids2 = [v.split("/")[-2] for v in vaspruns2]
icsd_ids3 = [v.split("/")[-2] for v in vaspruns3]

vaspruns_train = glob.glob("DFT-train/Ag-Au/*/vasprun.xml.polymlp")
vaspruns_test = glob.glob("DFT-test/Ag-Au/*/vasprun.xml.polymlp")

rep = PypolymlpRepository(mlp_paths=mlp_paths, verbose=True)

# If computational costs are not evaluated, the following must be activated.
# rep.calc_costs()

rep.extract_convex_polymlps(key="dataset/vasprun-*.xml.polymlp")
rep.calc_properties_binary_alloys(
    vaspruns_binary_prototypes=vaspruns1,
    vaspruns_element_prototypes1=vaspruns2,
    vaspruns_element_prototypes2=vaspruns3,
    icsd_ids_binary=icsd_ids1,
    icsd_ids_element1=icsd_ids2,
    icsd_ids_element2=icsd_ids3,
    vaspruns_train=vaspruns_train,
    vaspruns_test=vaspruns_test,
)
rep.generate_web_contents()
```
