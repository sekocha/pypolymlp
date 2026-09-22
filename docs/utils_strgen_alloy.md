# Generator of Substitutional (Alloy) Structures

## Generator of Random Substitutional Structures with Atomic Displacements and Cell Changes

> **Note**: This feature requires version 0.21.6 or later.

When random structures with atomic substitutions, atomic displacements, and cell changes are used to construct datasets, `pypolymlp` provides the `pypolymlp-structure` command-line interface with the `--substitution` option for generating substitutional alloy structures.

For example, consider substitutional structures of Sr(Zr,Ti)O3.
Suppose that the base structure is given by the following POSCAR:
```
> cat POSCAR
1.0
   7.873956509449926 0.000000000000000 0.000000000000000
   0.000000000000000 7.873956509449926 0.000000000000000
   0.000000000000000 0.000000000000000 7.907105922700000
 Sr  Zr  Ti  O
 8   4   4   24
Direct
    0.75 0.25 0.25
    (... skipped)
```
Random substitutions of Zr and Ti can be specified using the `--types 1 2` option, where atom types are identified by integer indices starting from zero.

Atomic displacements and cell changes can then be introduced into these substitutional structures using the structure-generation procedures implemented in `pypolymlp`.
The available procedures for generating random structures are described in [Generator of DFT random structures](utils_strgen.md).
All options for generating random structures can be combined with the `--substitution` option.

If 10 substitutional structures are generated and 20 sets of random atomic displacements and cell changes are introduced for each substitutional structure using the standard algorithm, the following command can be used:
```shell
pypolymlp-structure -p POSCAR --standard 20 --max_distance 1.0 --substitution 10 --types 1 2
```
In total, 200 structures are generated, and the corresponding POSCAR files are saved in the `poscars` directory.

If substitutions on multiple sublattices are required, the `--types` option can be specified multiple times, for example, as `--types 1 2 --types 3 4`.
In this example, atoms of types 1 and 2 are randomly substituted with each other, while atoms of types 3 and 4 are randomly substituted with each other.


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
