from pypolymlp.core.interface_openmx import parse_openmx
from pypolymlp.core.utils import split_ids_train_test
from pypolymlp.mlp_dev.pypolymlp import Pypolymlp

# Parse openmx data files.
datafiles = ["sample.md"]
structures, energies, forces = parse_openmx(datafiles)

# Split dataset into training and test datasets automatically.
n_data = len(energies)
train_ids, test_ids = split_ids_train_test(n_data, train_ratio=0.9)
train_structures = [structures[i] for i in train_ids]
test_structures = [structures[i] for i in test_ids]
train_energies = energies[train_ids]
test_energies = energies[test_ids]
train_forces = [forces[i] for i in train_ids]
test_forces = [forces[i] for i in test_ids]


polymlp = Pypolymlp()
polymlp.set_params(
    elements=["Ag", "C"],
    cutoff=8.0,
    model_type=3,
    max_p=2,
    gtinv_order=3,
    gtinv_maxl=(8, 8),
    reg_alpha_params=(-6, -3, 30),
    gaussian_params2=(1.0, 7.0, 7),
    atomic_energy=(0.0, 0.0),
)

polymlp.set_datasets_structures(
    train_structures=train_structures,
    test_structures=test_structures,
    train_energies=train_energies,
    test_energies=test_energies,
    train_forces=train_forces,
    test_forces=test_forces,
)
polymlp.run(verbose=True)
polymlp.save_mlp(filename="polymlp.yaml")
