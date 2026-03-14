# Interface to Phono3py Datasets

## MLP development using a parameter input file
```python
import numpy as np
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.load_parameter_file("polymlp.in")
polymlp.load_datasets()
polymlp.run(verbose=True)

polymlp.save_mlp(filename="polymlp.yaml")

params = polymlp.parameters
mlp_info = polymlp.summary
```

## MLP development from vasprun.xml files without using a parameter file
```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

"""
Parameters
----------
elements: Element species, (e.g., ['Mg','O'])
include_force: Considering force entries
include_stress: Considering stress entries
cutoff: Cutoff radius (Angstrom)
model_type: Polynomial function type
    model_type = 1: Linear polynomial of polynomial invariants
    model_type = 2: Polynomial of polynomial invariants
    model_type = 3: Polynomial of pair invariants
                    + linear polynomial of polynomial invariants
    model_type = 4: Polynomial of pair and second-order invariants
                    + linear polynomial of polynomial invariants
max_p: Order of polynomial function
feature_type: 'gtinv' or 'pair'
gaussian_params: Parameters for exp[- param1 * (r - param2)**2]
    Parameters are given as np.linspace(p[0], p[1], p[2]),
    where p[0], p[1], and p[2] are given by gaussian_params1
    and gaussian_params2.
reg_alpha_params: Parameters for penalty term in
    linear ridge regression. Parameters are given as
    np.linspace(p[0], p[1], p[2]).
gtinv_order: Maximum order of polynomial invariants.
gtinv_maxl: Maximum angular numbers of polynomial invariants.
    [maxl for order=2, maxl for order=3, ...]
atomic_energy: Atomic energies (in eV).
rearrange_by_elements: Set True if not developing special MLPs.
"""

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Mg','O'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,8],
    atomic_energy=[-0.00040000,-1.85321219],
)
train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```

## MLP development from phono3py.yaml.xz without using a parameter file

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    elements=['Ag','I'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,10],
    atomic_energy=[-0.19820116,-0.21203241],
)
train_yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
test_yaml = 'phono3py_params_wurtzite_AgI.yaml.xz'
train_ids = np.arange(20)
test_ids = np.arange(380,400)

polymlp.set_datasets_phono3py(
    train_yaml,
    test_yaml,
    train_ids=train_ids,
    test_ids=test_ids,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
```
