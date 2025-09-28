# MLP development for electronic free energy at finite temperatures

## Electronic free energy calculation from vasprun.xml files
```shell
> pypolymlp-utils --electron_vasprun */vasprun.xml --temp_max 2000 --temp_step 50

(`joblib` required.)
> pypolymlp-utils --electron_vasprun */vasprun.xml --temp_max 2000 --temp_step 50 --n_jobs -1
```

## MLP development
### At single temperature
```shell
> pypolymlp -i polymlp.in

# polymlp.in
dataset_type electron
data ./*/electron.yaml
temperature 500
include_force False
```
or
```python
yamlfiles = sorted(glob.glob("./*/electron.yaml"))
polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Ti"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8, 8],
    gaussian_params2=[0.0, 6.0, 7],
    atomic_energy=[0],
    reg_alpha_params=(-3, 1, 30),
)
polymlp.set_datasets_electron(yamlfiles, temperature=500)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml.500")
```

### At multiple temperatures
```python
yamlfiles = sorted(glob.glob("./*/electron.yaml"))
polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Ti"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=[8, 8],
    gaussian_params2=[0.0, 6.0, 7],
    atomic_energy=[0],
    reg_alpha_params=(-3, 1, 30),
)

temperatures = [0, 100, 200, 300, 400, 500]
for temp in temperatures:
    polymlp.set_datasets_electron(
        yamlfiles,
        temperature=temp,
        target="free_energy",
        verbose=True,
    )
    polymlp.run(verbose=True)
    polymlp.save_mlp(filename="polymlp.yaml." + str(temp))
```
