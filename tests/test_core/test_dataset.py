"""Tests of dataset."""

from pathlib import Path

# import numpy as np

# from pypolymlp.core.dataset import Dataset, DatasetList

cwd = Path(__file__).parent


def test_dataset():
    """Test for Dataset class."""


# def test_set_dataset_from_structures(structure_rocksalt):
#     """Test for set_dataset_from_structures."""
#     energies = [2.0, 1.0]
#     forces = [
#         np.random.random(structure_rocksalt.positions.shape),
#         np.random.random(structure_rocksalt.positions.shape),
#     ]
#     structures = [structure_rocksalt, structure_rocksalt]
#
#     element_order = ("Mg", "O")
#     dft1 = set_dataset_from_structures(
#         structures,
#         energies,
#         forces,
#         element_order=element_order,
#     )
#     forces1 = dft1.forces.reshape((2, 8, 3))
#     positions1 = [st.positions for st in dft1.structures]
#     elements1 = [st.elements for st in dft1.structures]
#     types1 = [st.types for st in dft1.structures]
#
#     element_order = ("O", "Mg")
#     dft2 = set_dataset_from_structures(
#         structures,
#         energies,
#         forces,
#         element_order=element_order,
#     )
#     order = [4, 5, 6, 7, 0, 1, 2, 3]
#     forces2 = dft2.forces.reshape((2, 8, 3))[:, order, :]
#     positions2 = [st.positions[:, order] for st in dft2.structures]
#     elements2 = [np.array(st.elements)[order] for st in dft2.structures]
#     types2 = [np.array(st.types) for st in dft2.structures]
#
#     np.testing.assert_allclose(forces1, forces2)
#     np.testing.assert_allclose(positions1, positions2)
#     np.testing.assert_equal(elements1, elements2)
#     np.testing.assert_equal(types1, types2)
