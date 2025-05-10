# How to use polymlp in other calculator tools

## LAMMPS
To use polynomial MLPs in LAMMPS, please use the [polymlp-lammps-package](https://github.com/sekocha/lammps-polymlp-package).
Molecular dynamics and other calculations can be performed following the standard LAMMPS documentation, except for the specification of `pair_style` and `pair_coeff`.

## ASE
To use polynomial MLPs in ASE, generate a calculator instance as shown below.
This calculator can be used for various calculations provided by ASE, in accordance with the ASE documentation.
```python
from pypolymlp.calculator.utils.ase_calculator import PolymlpASECalculator

calculator = PolymlpASECalculator(pot="polymlp.yaml")
```

## Phonopy
```python
from pypolymlp.calculator.properties import Properties
from pypolymlp.utils.phonopy_utils import phonopy_cell_to_structure

from phonopy import Phonopy

properties = Properties(pot="polymlp.yaml")

supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
ph = Phonopy(unitcell, supercell_matrix) # unitcell: PhonopyAtoms
ph.generate_displacements(distance=0.01)
supercells = ph.supercells_with_displacements
structures = [phonopy_cell_to_structure(cell) for cell in supercells]
_, forces, _ = self.prop.eval_multiple(structures)
forces = np.array(forces).transpose((0, 2, 1))
ph.forces = forces
ph.produce_force_constants()
```
