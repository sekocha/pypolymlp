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
