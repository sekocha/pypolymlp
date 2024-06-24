# Python API (MLP development)

## MLP development using polymlp.in
```python
import numpy as np
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.run(file_params='polymlp.in', verbose=True)

params_dict = polymlp.parameters
mlp_dict = polymlp.summary
```

## MLP development from vasprun.xml files without using polymlp.in
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
    ['Mg','O'],
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
```
or
```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

params = {
    'elements': ['Mg','O'],
    'cutoff' : 8.0,
    'model_type' : 3,
    'max_p' : 2,
    'gtinv_order' : 3,
    'gtinv_maxl' : [4,4],
    'gaussian_params2' : [0.0, 7.0, 8],
    'atomic_energy' : [-0.00040000,-1.85321219],
}

polymlp = Pypolymlp()
polymlp.set_params(params=params)

train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
polymlp.run(verbose=True)
```

## MLP development from phono3py.yaml.xz without using polymlp.in

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    ['Ag','I'],
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
train_energy_dat = 'energies_ltc_wurtzite_AgI_fc3-forces.dat'
test_energy_dat = 'energies_ltc_wurtzite_AgI_fc3-forces.dat'
train_ids = np.arange(20)
test_ids = np.arange(380,400)

polymlp.set_datasets_phono3py(
    train_yaml,
    test_yaml,
    train_energy_dat=train_energy_dat,
    test_energy_dat=test_energy_dat,
    train_ids=train_ids,
    test_ids=test_ids,
)
polymlp.run(verbose=True)
```

When energy values are read from phono3py.yaml.xz, train_energy_dat and test_energy dat are not required as follows.
```python
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
```


## MLP development using displacements and forces

```python
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

polymlp = Pypolymlp()
polymlp.set_params(
    ['Ag','I'],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[4,4],
    gaussian_params2=[0.0,7.0,10],
    atomic_energy=[-0.19820116,-0.21203241],
)

'''
Parameters in polymlp.set_datasets_displacements
-------------------------------------------------
train_disps: (n_train, 3, n_atoms)
train_forces: (n_train, 3, n_atoms)
train_energies: (n_train)
test_disps: (n_test, 3, n_atom)
test_forces: (n_test, 3, n_atom)
test_energies: (n_test)

structure_without_disp: supercell structure without displacements, dict
(keys)
- 'axis': (3,3), [a, b, c]
- 'positions': (3, n_atom) [x1, x2, ...]
- 'n_atoms': [4, 4]
- 'elements': Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
- 'types': Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
- 'volume': 64.0 (ang.^3)
'''

polymlp.set_datasets_displacements(
    train_disps,
    train_forces,
    train_energies,
    test_disps,
    test_forces,
    test_energies,
    structure_without_disp,
)
polymlp.run(verbose=True)
```

## From multiple sets of vasprun.xml files

```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

params = {
    'elements': ['Mg','O'],
    'cutoff' : 8.0,
    'model_type' : 3,
    'max_p' : 2,
    'gtinv_order' : 3,
    'gtinv_maxl' : [4,4],
    'gaussian_params2' : [0.0, 7.0, 8],
    'atomic_energy' : [-0.00040000,-1.85321219],
}

polymlp = Pypolymlp()
polymlp.set_params(params=params)

train_vaspruns1 = glob.glob('vaspruns/train1/vasprun-*.xml.polymlp')
train_vaspruns2 = glob.glob('vaspruns/train2/vasprun-*.xml.polymlp')
test_vaspruns1 = glob.glob('vaspruns/test1/vasprun-*.xml.polymlp')
test_vaspruns2 = glob.glob('vaspruns/test2/vasprun-*.xml.polymlp')
polymlp.set_multiple_datasets_vasp(
    [train_vaspruns1, train_vaspruns2],
    [test_vaspruns1, test_vaspruns2]
)

#polymlp.run(verbose=True, sequential=False)
polymlp.run(verbose=True, sequential=True)
```
