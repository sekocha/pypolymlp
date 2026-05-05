# Finite-Temperature Elastic Constants

> **Note**: Requires version 0.19.9 or later.

Coming soon.

## MLP for SSCHA Elastic Constant Calculations

To perform accurate SSCHA calculations for target compounds and structures across a range of isotropic volume changes, shear deformation and temperatures, it is necessary to use either general-purpose polynomial MLPs or on-the-fly polynomial MLPs.
These models should enable accurate property evaluations for structures with large atomic displacements generated from the effective harmonic Hamiltonian in SSCHA calculations.

When SSCHA calculations are carried out for a single compound across a range of axis changes and temperatures, developing an on-the-fly MLP can be a suitable option, as it can provide higher accuracy for the target system than a general-purpose MLP.

See [Development of On-the-fly MLP](../tutorial_onthefly.md) for more details.
