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

## Generalized Stacking Fault Energy
### Calculations for the entire set of points
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_poscars("POSCAR")
energies = polymlp.run_gsfe(
    disp1=(1, -1, 0),
    disp2=(1, 1, -2),
    glide_plane=(1, 1, 1),
    n_layers=2,
    n_points=5,
    filename="gsfe.dat",
)
```
### Single-point Calculation
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_poscars("POSCAR")
energy = polymlp.run_gsfe(
    disp1=(1, -1, 0),
    disp2=(1, 1, -2),
    glide_plane=(1, 1, 1),
    n_layers=2,
    frac1=0.25,
    frac2=0.3,
)
energy0 = polymlp.run_gsfe(
    disp1=(1, -1, 0),
    disp2=(1, 1, -2),
    glide_plane=(1, 1, 1),
    n_layers=2,
    frac1=0.0,
    frac2=0.0,
)

# Excess energy in J/m^2
excess = energy - energy0
```

## Transformation path
```python
from pypolymlp.calculator.compute_transformation import PolymlpTransformation

# Transformation path along fixed angles
trans = PolymlpTransformation(unitcell, prop, verbose=False)
trans.set_supercell(
    disp1=(1, 0, 0),
    disp2=(0, 1, 0),
    glide_plane=(0, 0, 1),
    n_layers=3,
)

trans.run_fix_angle(
    degs_min=90, degs_max=95, degs_int=1, axis1=0, axis2=2, gtol=1e-4
)
trans.save(filename="path.dat")

# Transformation path along shift of upper half of supercell.
trans = PolymlpTransformation(unitcell, prop, verbose=False)
trans.set_supercell(
    disp1=(1, 0, 0),
    disp2=(0, 1, 0),
    glide_plane=(0, 0, 1),
    n_layers=4,
)
trans.run_fix_shift(
    max_shift_frac=0.5,
    n_points=5,
    axis_shift=0,
    axis_normal_shift=2,
    gtol=1e-4,
)
trans.save(filename="path.dat")
```
