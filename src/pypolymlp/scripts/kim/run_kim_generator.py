"""Script for converting polymlp to KIM format.

Usage
-----
(For single model)
python3 run_kim_generator.py polymlp.yaml

(For hybrid model)
python3 run_kim_generator.py polymlp.yaml.1 polymlp.yaml.2
"""

import sys
from pathlib import Path

import numpy as np

from pypolymlp.api.pypolymlp_utils import PypolymlpUtils

pot = sys.argv[1:]

user_id = "b3113743-4f85-48da-86e1-85acf6bb3388"
author = "Seko"
project_id = np.random.randint(10**11, 10**12)
project_version = 0
path_license = str(Path(__file__).parent / "LICENSE")

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
    path_license=path_license,
)
