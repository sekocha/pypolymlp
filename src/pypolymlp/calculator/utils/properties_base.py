"""Base class for calculating properties."""

from abc import ABC, abstractmethod

from pypolymlp.core.data_format import PolymlpStructure


class PropertiesBase(ABC):
    """Base class for calculating properties."""

    def __init__(self):
        """Init method."""
        self._elements = None

    @abstractmethod
    def eval(self, st: PolymlpStructure, use_openmp: bool = True):
        """Evaluate properties for a single structure."""
        pass

    @abstractmethod
    def eval_multiple(self, structures: list[PolymlpStructure]):
        """Evaluate properties for multiple structures."""
        pass

    @abstractmethod
    @property
    def elements(self):
        """Return elements."""
        pass
