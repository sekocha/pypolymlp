# Compression of `vasprun.xml` Files

When using datasets from VASP, `vasprun.xml` files can serve as input files for the polynomial MLP development.
However, most of the data in `vasprun.xml` files is not required for MLP development.
When parsing a large number of original `vasprun.xml` files before calculating structural features and estimating MLP coefficients, it can take a significant amount of time due to the size and number of the files.
To address this, `pypolymlp` supports removing unnecessary content from `vasprun.xml` files and generating compact versions of these files.
The `--vasprun_compress` option generates compact files for multiple `vasprun.xml` files as follows:

```shell
> pypolymlp-utils --vasprun_compress vaspruns/vasprun-*.xml
```

The compressed files are generated as `vasprun.xml.polymlp` for each original `vasprun.xml` file.
The generated `vasprun.xml.polymlp` files can be used for the MLP developement by specififying these files as datasets in the input file or variables of the Python API.

```shell
data vaspruns/vasprun-*.xml.polymlp
```
