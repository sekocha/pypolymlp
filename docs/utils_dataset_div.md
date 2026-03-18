# Automatic division of DFT dataset

```shell
> pypolymlp-utils --auto_dataset dataset1/*/vasprun.xml dataset2/*/vasprun.xml
> cat polymlp.in.append >> polymlp.in
```
A given DFT dataset is automatically divided into some sets, depending on the values of the energy, the forces acting on atoms, and the volume.
A generated file "polymlp.in.append" can be appended in your polymlp.in, which will be used for developing MLPs.
Datasets identified with "train1" and "test1" are composed of structures with low energy and small force values.
The predictive power for them is more important than the other structures for the successive calculations using polynomial MLPs, so the prediction errors for "train1" and "test1" datasets should be accuracy measures for polynomial MLPs.
