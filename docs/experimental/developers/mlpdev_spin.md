# Development of Spin-Dependent MLPs

> **Note**: Version 0.19.2 or later is required.
Only datasets generated from VASP are supported.

Let us consider a system in which elements have collinear up and down spins, such as an antiferromagnetic system.
When it is essential to explicitly account for spin collinear configurations, it is natural to perform collinear spin-polarized DFT calculations to generate training and test datasets for such a system.
To develop MLPs for such a system, a framework for spin-configuration-dependent MLPs may be useful.


In the procedure for developing spin-configuration-dependent polynomial MLPs implemented in `pypolymlp`, elements with up and down spins are treated as distinct species.
For instance, when the element Fe exhibits both up and down spins in the dataset, `pypolymlp` may regard the target system as a binary system consisting of Fe with up spin and Fe with down spin.


## 1. Generation of Datasets from Spin-Polarized DFT Calculations

Datasets must first be prepared using spin-polarized DFT calculations.
`pypolymlp` supports the development of spin-dependent MLPs only with datasets generated from VASP.

It is important that, when performing spin-polarized DFT calculations to generate datasets for use in `pypolymlp`, the ordering of elements with up and down spins in each structure is kept consistent, with elements having up spins listed before those having down spins.
In the following `POSCAR` example, the first type of Fe must correspond to up spins, and the second type of Fe must correspond to down spins.
This ordering of spin assignments must be kept consistent across all structures in the dataset.

```
FCC
 1.0
   4.000 0.000 0.000
   0.000 4.000 0.000
   0.000 0.000 4.000
   Fe   Fe
   2    2
Direct
   0.000 0.000 0.000
   0.000 0.500 0.500
   0.500 0.000 0.500
   0.500 0.500 0.000
```

## 2. Development of Spin-Dependent MLPs

### Command-Line Interface
To develop spin-polarized MLPs, the `pypolymlp` command with the `-i` option can be used in the same way as for developing standard MLPs.

```shell
> pypolymlp -i polymlp.in
```

To enable the distinction between up and down spins in polynomial MLP models, the `enable_spins` tag must be included in the input file, as shown below.

```
> cat polymlp.in
feature_type pair
cutoff 6.0
n_gaussians 9
model_type 2
max_p 3

include_force True
include_stress True
reg_alpha_params -4 1 6

n_type 1
elements Fe
atomic_energy -3.37684106
train_data data-vasp-spin-Fe/vaspruns/train/*.xml True 1.0
test_data data-vasp-spin-Fe/vaspruns/test/*.xml True 1.0

enable_spins True
```

To enable spin configurations in binary or ternary systems, set the `enable_spins` variable to multiple boolean values, for example:
```
n_type 2
elements Fe Ti
enable_spins True False
```

### Python API

To develop spin-polarized MLPs using the Python API, set the `enable_spins` parameter in both `set_params` and `append_hybrid_params`.

```python
import numpy as np
import glob
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp


polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Fe"],
    cutoff=6.0,
    model_type=2,
    max_p=3,
    feature_type="pair",
    reg_alpha_params=(-4, 1, 6),
    n_gaussians=9,
    atomic_energy=(-3.37684106,),
    enable_spins=(True,),
)

train_vaspruns = glob.glob('vaspruns/train/vasprun-*.xml.polymlp')
test_vaspruns = glob.glob('vaspruns/test/vasprun-*.xml.polymlp')
polymlp.set_datasets_vasp(train_vaspruns, test_vaspruns)
polymlp.run()
polymlp.save_mlp(filename="polymlp.yaml")
```
