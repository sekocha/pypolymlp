"""Tests of KIM utility functions."""

import glob
import shutil
from pathlib import Path

from pypolymlp.utils.kim_utils import convert_polymlp_to_kim_model

cwd = Path(__file__).parent
path_file = str(cwd) + "/../files/"


def test_convert_polymlp_to_kim_model():
    """Test convert_polymlp_to_kim_model."""

    user_id = "b3113743-4f85-48da-86e1-85acf6bb3388"
    citation1 = {
        "article-number": "{011101}",
        "author": "Seko, Atsuto",
    }
    citations = [citation1]
    convert_polymlp_to_kim_model(
        path_file + "polymlp.yaml.MgO",
        author="Seko",
        performance_level=1,
        project_id=123,
        project_version=2,
        model_driver="Polymlp__MD_367995833009_000",
        content_origin="pypolymlp",
        contributor_id=user_id,
        developer=[user_id],
        maintainer_id=user_id,
        citations=citations,
    )
    for d in glob.glob("Polymlp_Seko_*_MgO__MO_000000000123_002"):
        shutil.rmtree(d)
