# Automated Division of DFT Datasets

> **Note**: Version 0.19.1 or later.

`pypolymlp` supports automatic dataset divisions into some subsets, depending on the values of the energy and the forces acting on atoms.
In addition, the weight values are automatically assigned to dataset entries according to the values of the energy and the forces.


# Command-line Interface
```shell
> pypolymlp-utils --auto_dataset dataset1/*/vasprun.xml dataset2/*/vasprun.xml --elements Si --n_divide 6
```

`polymlp_datasets`

A generated file "polymlp.in.append" can be appended in your polymlp.in, which will be used for developing MLPs.
Datasets identified with "train1" and "test1" are composed of structures with low energy and small force values.
The predictive power for them is more important than the other structures for the successive calculations using polynomial MLPs, so the prediction errors for "train1" and "test1" datasets should be accuracy measures for polynomial MLPs.

```
> cat polymlp.in.append >> polymlp.in
```

# Python API
```python

```
