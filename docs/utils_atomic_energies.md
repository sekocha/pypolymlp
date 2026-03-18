## Atomic energies
(Experimental: Only for VASP calculations using PBE and PBEsol functionals)

```shell
> pypolymlp-utils --atomic_energy_elements Mg O --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula MgO --atomic_energy_functional PBE
> pypolymlp-utils --atomic_energy_formula Al2O3 --atomic_energy_functional PBEsol
```

A standard output can be appended in your polymlp.in, which will be used for developing MLPs.
The polynomial MLP has no constant term, which means that the energy for isolated atoms is set to zero.
The energy values for the isolated atoms must be subtracted from the energy values for structures.
