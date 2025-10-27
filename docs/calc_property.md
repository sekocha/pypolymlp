# Property calculations using polynomial MLP

The energy, atomic forces, and stress tensor can be calculated through the command-line interface or the Python API.
If multiple structure files are provided as input, property calculations are compatible with OpenMP support.
If a single structure is provided, OpenMP support is still available, but the computation may be slower for small systems.

## Using command line interface
Given a polynomial MLP file `polymlp.yaml` and a structure specified by `POSCAR`, the energy, atomic forces, and stress tensor can be calculated using the following command.

```shell
(For single structure)
> pypolymlp-calc --properties --pot polymlp.yaml --poscars POSCAR

(For multiple structures)
> pypolymlp-calc --properties --pot polymlp.yaml --poscars */POSCAR
> pypolymlp-calc --properties --pot polymlp.yaml --vaspruns vaspruns/vasprun.xml.polymlp.*
```

When using a hybrid polynomial MLP, multiple MLP files should be given for `--pot` option.
```shell
--pot polymlp.yaml*
or
--pot polymlp.yaml.1 polymlp.yaml.2
```

After the property calculations are completed, the results are saved as NumPy objects:

- `polymlp_energies.npy` with shape `(n_structure)` for the energies, in units of eV/supercell.
- `polymlp_forces.npy` with shape `(n_structure, 3, n_atom)` for the atomic forces, in units of eV/angstroms, if all structures have the same number of atoms.
  If the structures contain different numbers of atoms, multiple files such as `polymlp_forces_00001.npy` are generated for each structure.
- `polymlp_stress_tensor.npy` with shape `(n_structure, 6)` for the stress tensor, in the order xx, yy, zz, xy, yz, zx, and in units of eV/cell.

## Using Python API

Python API for calculating properties can be initialized as follows.
```python
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
polymlp = PypolymlpCalc(pot="polymlp.yaml")

# For legacy polymlp.lammps
polymlp = PypolymlpCalc(pot="polymlp.lammps")
```

### For single structure
Given a polynomial MLP file `polymlp.yaml` and a structure specified by `POSCAR`, the energy, atomic forces, and stress tensor can be calculated as follows.
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
"""
energy: unit: eV/supercell
forces: unit: eV/angstrom (3, n_atom)
stress: unit: eV/supercell: (6) in the order of xx, yy, zz, xy, yz, zx
"""
polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_structures_from_files(poscars='POSCAR')
energies, forces, stresses = polymlp.eval()
```

If a structure is provided without using a `POSCAR` file, it should be supplied as a `PolymlpStructure` instance.
```python
import numpy as np
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.data_format import PolymlpStructure
"""
structure: PolymlpStructure format.
Attributes:
- axis: (3,3), [a, b, c]
- positions: (3, n_atom) [x1, x2, ...]
- n_atoms: [4, 4]
- elements: Element list (e.g.) ['Mg','Mg','Mg','Mg','O','O','O','O']
- types: Atomic type integers (e.g.) [0, 0, 0, 0, 1, 1, 1, 1]
"""
# structure = Poscar('POSCAR').structure
polymlp = PypolymlpCalc(pot="polymlp.yaml")
energies, forces, stresses = polymlp.eval(structure)
```

### For multiple structures (Compatible with OPENMP support)
If multiple structure files are provided as input, property calculations are compatible with OpenMP support.
The properties for these structures can be calculated by specifying either `POSCAR` files or `PolymlpStructure` instances.

```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

"""
energy_all: unit: eV/supercell (n_str)
forces_all: unit: eV/angstrom (n_str, 3, n_atom)
stress_all: unit: eV/supercell: (n_str, 6)
                                in the order of xx, yy, zz, xy, yz, zx
"""
polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_structures_from_files(poscars=["POSCAR1", "POSCAR2", "POSCAR3"])
energy_all, forces_all, stress_all = polymlp.eval()
```
or
```python
structures = [structure1, structure2, structure3]
energy_all, forces_all, stress_all = polymlp.eval(structures)
```

### Hybrid polynomial MLP
When using a hybrid polynomial MLP, multiple MLP files should be given as a list.
```python
polymlps = ['polymlp.yaml.1', 'polymlp.yaml.2']
polymlp_calc = PypolymlpCalc(pot=polymlps)
```

### From phonopy structure instances
```python
import numpy as np
from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
"""
energy_all: unit: eV/supercell (n_str)
forces_all: unit: eV/angstrom (n_str, 3, n_atom)
stress_all: unit: eV/supercell: (n_str, 6)
                                in the order of xx, yy, zz, xy, yz, zx
"""
# phonopy.generate_displacements(distance=0.01)
# supercells = ph.supercells_with_displacements

polymlp = PypolymlpCalc(pot="polymlp.yaml")
polymlp.load_phonopy_structures(supercells)
energy_all, forces_all, stress_all = polymlp.eval()
```

- Conversion of a phonopy cell instance into a structure dictionary
```python
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure
structure = phonopy_cell_to_structure(cell_phonopy)
```
