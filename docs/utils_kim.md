# Generation of an OpenKIM Portable Model for Polynomial MLP

Polynomial MLPs can be converted into portable models in OpenKIM that are compatible with the [OpenKIM PolyMLP model driver](https://openkim.org/id/PolyMLP__MD_367995833009_000). Portable models can be generated using the Python API `PypolymlpUtils`. An example of converting a polynomial MLP into a portable model is shown below.

```python
import sys
import numpy as np
from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

# variable "pot" must be string for single polymlp model
#   or list of string for hybrid polymlp model
pot = sys.argv[1:]
user_id = "b3113743-4f85-48da-86e1-85acf6bb3388"
author = "Seko"

project_id = np.random.randint(10**11, 10**12)
project_version = 0

recordkey = ("MO", str(project_id).zfill(12), str(project_version).zfill(3))
recordkey = "_".join(recordkey)

citation1 = {
    "article-number": "{011101}",
    "author": "Seko, Atsuto",
    "doi": "10.1063/5.0129045",
    "journal": "{J. Appl. Phys.}",
    "eissn": "{1089-7550}",
    "issn": "{0021-8979}",
    "orcid-numbers": "{Seko, Atsuto/0000-0002-2473-3837}",
    "recordkey": recordkey + "a",
    "recordprimary": "recordprimary",
    "recordtype": "article",
    "unique-id": "{WOS:000908391700010}",
    "title": "{Tutorial: Systematic development of polynomial machine learning potentials for elemental and alloy systems}",
    "volume": "{133}",
    "number": "{1}",
    "year": "{2023}",
    "month": "{Jan}",
}
citations = [citation1]

polymlp = PypolymlpUtils()
polymlp.generate_kim_model(
    pot,
    author=author,
    performance_level=1,
    project_id=project_id,
    project_version=project_version,
    model_driver="Polymlp__MD_367995833009_000",
    content_origin="pypolymlp",
    contributor_id=user_id,
    developer=[user_id],
    maintainer_id=user_id,
    citations=citations,
)
```
