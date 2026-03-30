# MLP Development for Substitutionally Disordered Structures

> **Note**: Requires version 0.19.3 or later.

## 1. DFT Calculations for Substitutional Configurations on a Lattice with Random Structural Changes

Given a lattice and a set of elements, datasets covering the target range of substitutional atomic configurations, along with their derivatives involving random structural changes, are first generated using DFT calculations.
These structural changes include atomic displacements, volume changes, and shear deformations.

## 2. Standard MLP Development

Using datasets containing substitutional atomic configurations and their derivatives, MLPs can be developed by distinguishing all atomic species through a standard procedure.


## 3. Property Calculations for Substitutionally Disordered States Using MLPs

By defining element occupation probabilities on the lattice or sublattices, MLPs for disordered states represented by these probabilities can be developed using the Python API.

Before developing MLPs for disordered states, properties of `n_samples` disordered structures are calculated using the standard MLP.
The properties of disordered structures are calculated as averages over substitutional a
tomic configurations represented by the occupation probabilities.
Each substitutionally disordered structure includes random structural changes from the lattice, such as atomic displacements, volume changes, and shear deformations.


```python
from pypolymlp.postproc.disorder import PolymlpDisorder

occupancy = [
    [("Sr", 2/3), ("Cu", 1/3)],
    [("Cu", 2/3), ("Sr", 1/3)],
    [("O", 1.0)],
]

disorder = PolymlpDisorder(
    occupancy=occupancy,
    pot="polymlp.yaml",
    lattice="POSCAR",
    supercell_size=(3, 3, 3),
    verbose=True,
)
disorder.set_displaced_lattices(n_samples=3000, max_distance=1.0)
disorder.eval_random_properties(etol=1e-4)
disorder.save_properties(path="polymlp_disorder")

# If properties are used in successive Python codes,
# the following attributes can be used.
energies, forces, stresses = disorder.properties
```

- **Example for paramagnetic system**

```python
from pypolymlp.postproc.disorder import PolymlpDisorder


occupancy = [[(("Fe", 0), 1/2), (("Fe", 1), 1/2)]]

disorder = PolymlpDisorder(
    occupancy=occupancy,
    pot="polymlp.yaml",
    lattice_poscar="POSCAR",
    supercell_size=(3, 3, 3),
    verbose=True,
)
disorder.set_displaced_lattices(n_samples=3000, max_distance=0.8)
disorder.eval_random_properties(etol=1e-4)
disorder.save_properties(path="polymlp_disorder")
```


## 4. MLP Developement for Substitutionally Disordered State

Using datasets generated from MLP calculations for disordered structures, MLPs for the disordered state can be developed using the Python API.

```python
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

yamlfiles = glob.glob("polymlp_disorder/polymlp_disorder_*.yaml")
polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Sr", "Cu", "O"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8, 8],
    n_gaussians=10,
    atomic_energy=(0, 0, 0),
    reg_alpha_params=(-5, 5, 30),
)
polymlp.set_datasets_property_yamls(yamlfiles)
polymlp.run()
polymlp.save_mlp(filename="polymlp.yaml.disorder")
```

- **Example for paramagnetic system**

```python
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

yamlfiles = glob.glob("polymlp_disorder/polymlp_disorder_*.yaml")

polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Fe"],
    cutoff=6.0,
    model_type=4,
    max_p=2,
    gtinv_order=4,
    gtinv_maxl=[8, 4, 2],
    n_gaussians=12,
    atomic_energy=(0.0, ),
    reg_alpha_params=(-5, 5, 30),
)
polymlp.set_datasets_property_yamls(yamlfiles)
polymlp.run()
polymlp.save_mlp(filename="polymlp.yaml.disorder")
```
