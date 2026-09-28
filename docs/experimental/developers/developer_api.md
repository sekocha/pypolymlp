# Python API for pypolymlp developers

## Generation of Surface Slab Models
```python
import numpy
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils
from pypolymlp.core.interface_vasp import Poscar

st = Poscar("POSCAR").structure

utils = PypolymlpUtils()
sup = utils.generate_slab_model(
    structure=st,
    direction1=(1, 0, 0),
    direction2=(0, 1, 0),
    direction3=(0, 0, 1),
    n_layers=8,
    end_frac=None,
)
utils.write_poscar_file(sup, filename="POSCAR.slab.001")

sup = utils.generate_slab_model(
    structure=st,
    direction1=(1, 0, 0),
    direction2=(0, 1, -1),
    direction3=(0, 1, 1),
    n_layers=6,
    end_frac=0.0,
)
utils.write_poscar_file(sup, filename="POSCAR.slab.011")

sup = utils.generate_slab_model(
    structure=st,
    direction1=(1, -1, 0),
    direction2=(1, 1, -2),
    direction3=(1, 1, 1),
    n_layers=4,
    end_frac=None,
)
utils.write_poscar_file(sup, filename="POSCAR.slab.111")
```
