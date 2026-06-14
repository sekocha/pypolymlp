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
        Distance list. shape=(n_atom, n_neighbor).
        """
        return self._obj.get_distances()
