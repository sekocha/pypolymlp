# Generator of Substitutional (Alloy) Structures

## Generator of Random Substitutional Structures with Atomic Displacements and Cell Changes

Coming soon.

## Generator of Derivative Structures with Atomic Displacements and Cell Changes

> **Note**: Requires [`pyclupan`](https://github.com/sekocha/pyclupan) and `graphillion`.

Random structures can also be generated from derivative structures, which are defined as nonequivalent substitutional configurations on a given lattice.
The derivative structures can be enumerated using `pyclupan`, and all or a subset of them can be used as base structures to introduce atomic displacements and cell changes.

1. **Enumerate Derivative Structures**

For information on how to enumerate derivative structures, please refer to [Enumeration of Derivative Structures](https://github.com/sekocha/pyclupan/blob/develop/docs/calc_derivative.md).
As a result of this enumeration, a summary file named `pyclupan_derivatives.yaml` containing the derivative structures is generated.

2. **Generate Derivative Structures with Atomic Displacements and Cell Changes**

Using the `pyclupan_derivatives.yaml` file, structures with atomic displacements and cell changes can be generated.
`pyclupan` provides a command-line interface, `pyclupan-sample`, where the `--displacements` option is used to introduce atomic displacements and cell changes.

In the following example, after enumerating binary derivative structures, 10 derivative structures are randomly sampled from the entire set of enumerated derivative structures, and 10 structures with atomic displacements and cell changes are generated for each base derivative structure.
                                                        
```shell                                                
pyclupan-sample --yaml pyclupan_derivatives.yaml --method random --element_strings Ag Au -n 10 --displacements 0.1 --n_disps 10
```
Structures with atomic displacements and cell changes are generated in the `poscars_disps` directory.

If multiple supercell sizes or supercell expansions are used in the derivative structure enumeration, a wildcard format can be used to specify multiple files, as shown below:

```shell
pyclupan-sample --yaml pyclupan_derivatives_*.yaml --method random --element_strings Ag Au -n 10 --displacements 0.1 --n_disps 10
```
