"""Tests of API for generating web contents."""

import glob
import shutil
from pathlib import Path

from pypolymlp.calculator.auto_repository.web import WebContents

cwd = Path(__file__).parent
path_file = str(cwd) + "/files/"


def test_web():
    """Test web content generation."""
    web = WebContents(path_prediction=path_file + "others/repository/Ag-Au-2026-03-10")
    web.run()
    for path in glob.glob("web-Ag-Au*"):
        shutil.rmtree(path)
