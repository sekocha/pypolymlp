"""API neighbor functions using C++ library."""

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.cxx.lib import libmlpcpp


class Neighbor:
    """Calculate neighbor atoms."""

    def __init__(self, structure: PolymlpStructure, cutoff: float = 6.0):
        """Init method."""
        if structure.positions_cartesian is None:
            structure.positions_cartesian = structure.axis @ structure.positions

        n_type = len(set(structure.types))
        self._obj = libmlpcpp.Neighbor(
            structure.axis,
            structure.positions_cartesian,
            structure.types,
            n_type,
            cutoff,
        )

    @property
    def distances(self):
        """Return distances.

        Return
        ------
        Distance list. shape=(n_atom, n_type, n_neighbor_i).
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_distances()

    @property
    def differences(self):
        """Return Cartesian vector between atom and neighbor atom.

        Return
        ------
        Cartesian difference vector. shape=(n_atom, n_type, n_neighbor_i, 3).
          Calculate positions[j] - positions[i] for central atom i and neighbor atom j.
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_differences()

    @property
    def neighbor_atoms(self):
        """Return neighbor atom indices.

        Return
        ------
        Neighbor atom indices. shape=(n_atom, n_type, n_neighbor).
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_neighbor_indices()


class NeighborHalf:
    """Calculate a half list of neighbor atoms."""

    def __init__(
        self,
        structure: PolymlpStructure,
        cutoff: float = 6.0,
        use_openmp: bool = True,
    ):
        """Init method."""
        if structure.positions_cartesian is None:
            structure.positions_cartesian = structure.axis @ structure.positions

        self._obj = libmlpcpp.NeighborHalf(
            structure.axis,
            structure.positions_cartesian,
            cutoff,
            use_openmp,
        )

    @property
    def differences(self):
        """Return Cartesian vector between atom and neighbor atom.

        Return
        ------
        Cartesian difference vector. shape=(n_atom, n_type, n_neighbor_i, 3).
          Calculate positions[j] - positions[i] for central atom i and neighbor atom j.
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_differences()

    @property
    def neighbor_atoms(self):
        """Return neighbor atom indices.

        Return
        ------
        Neighbor atom indices. shape=(n_atom, n_type, n_neighbor).
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_neighbor_indices()


class NeighborFull:
    """Calculate neighbor atoms."""

    def __init__(self, structure: PolymlpStructure, cutoff: float = 6.0):
        """Init method."""
        if structure.positions_cartesian is None:
            structure.positions_cartesian = structure.axis @ structure.positions

        self._types = structure.types
        self._n_type = len(set(structure.types))
        self._obj = libmlpcpp.NeighborFull(
            structure.axis,
            structure.positions_cartesian,
            cutoff,
        )

    @property
    def distances(self):
        """Return distances.

        Return
        ------
        Distance list. shape=(n_atom, n_type, n_neighbor_i).
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_distances(self._n_type, self._types)

    @property
    def differences(self):
        """Return Cartesian vector between atom and neighbor atom.

        Return
        ------
        Cartesian difference vector. shape=(n_atom, n_type, n_neighbor_i, 3).
          Calculate positions[j] - positions[i] for central atom i and neighbor atom j.
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_differences(self._n_type, self._types)

    @property
    def neighbor_atoms(self):
        """Return neighbor atom indices.

        Return
        ------
        Neighbor atom indices. shape=(n_atom, n_type, n_neighbor).
          Array indices correspond to (central atom i, atom type of neighboring atom j).
        """
        return self._obj.get_neighbor_indices(self._n_type, self._types)
