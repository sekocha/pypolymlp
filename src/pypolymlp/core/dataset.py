"""Dataclass of parameters for dataset."""

import glob
from dataclasses import dataclass
from typing import Literal, Optional

from pypolymlp.core.utils import split_train_test, strtobool


@dataclass
class Dataset:
    """Dataclass of parameters for dataset.

    Parameters
    ----------
    name: Name of dataset.
    dataset_type: Dataset type.
    string_list: String list for specifying dataset files, include_force, and weight.
    location: String for specifying dataset files.
    files: Dataset files.
    include_force: Consider force or not.
    weight: Weight for dataset.
    split: Dataset is obtained by splitting into training and test datasets.
    """

    dataset_type: Literal["vasp", "sscha", "electron", "phono3py"] = "vasp"
    string_list: Optional[str] = None
    location: Optional[str] = None
    files: Optional[list] = None
    include_force: bool = True
    weight: float = 1.0
    name: Optional[str] = None
    split: bool = True
    prefix: Optional[str] = None
    energy_dat: Optional[str] = None

    def __post_init__(self):
        """Post-init method."""
        if self.string_list is None and self.location is None and self.files is None:
            raise RuntimeError("All of string_list, location and files not given.")
        if self.string_list is not None:
            self._initialize_from_string()
        self._set_files_from_location()

    def _initialize_from_string(self):
        """Initialize from string input."""
        self.name = self.string_list[0]
        self.location = self.string_list[0]
        if len(self.string_list) > 1:
            self.include_force = strtobool(self.string_list[1])
        if len(self.string_list) > 2:
            self.weight = float(self.string_list[2])

    def _set_files_from_location(self):
        """Set files from location."""
        if self.location is not None:
            if not self.dataset_type == "phono3py":
                if self.prefix is None:
                    self.files = sorted(glob.glob(self.location))
                else:
                    self.files = sorted(glob.glob(self.prefix + "/" + self.location))

    def split_train_test(self, train_ratio: float = 0.9):
        """Split files in dataset into training and test datasets."""
        files_train, files_test = split_train_test(self.files, train_ratio=train_ratio)
        train = Dataset(
            dataset_type=self.dataset_type,
            string_list=self.string_list,
            location=self.location,
            files=files_train,
            include_force=self.include_force,
            weight=self.weight,
            name=self.name,
            split=True,
            prefix=self.prefix,
        )
        test = Dataset(
            dataset_type=self.dataset_type,
            string_list=self.string_list,
            location=self.location,
            files=files_test,
            include_force=self.include_force,
            weight=self.weight,
            name=self.name,
            split=True,
            prefix=self.prefix,
        )
        return train, test
