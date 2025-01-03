"""API class for post-calculations using systematic SSCHA results."""

from pypolymlp.calculator.sscha.sscha_properties import SSCHAProperties


class PolymlpSSCHAPost:
    """API class for post-calculations using systematic SSCHA results."""

    def __init__(
        self,
        verbose: bool = False,
    ):
        """Init method."""
        self._verbose = verbose

    def compute_thermodynamic_properties(
        self,
        yamlfiles: list[str],
        filename: str = "sscha_properties.yaml",
    ):
        """Calculate thermodynamic properties from SSCHA results."""
        sscha = SSCHAProperties(yamlfiles, verbose=self._verbose)
        sscha.run()
        sscha.save_properties(filename=filename)
