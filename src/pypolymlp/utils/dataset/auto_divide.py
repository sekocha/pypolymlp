#!/usr/bin/env python
import numpy as np

from pypolymlp.core.interface_vasp import parse_vaspruns
from pypolymlp.utils.dataset.dataset_utils import (  # split_two_datasets,
    copy_vaspruns,
    split_three_datasets,
)

#    suffix1 = ele + '/finished/'
#    dir1 = prefix + '2019-n10000-pre/2-ideal/k70/' + suffix1
#    dir1 = prefix + '2019-n10000/' + suffix1
#    dir1 = prefix + '2021-low-density-finished/' + suffix1
#    dir1 = prefix + '2020-n3000-large-d/' + suffix1
#    dir1 = prefix + '2020-n3000-pending/' + suffix1
#    dir1 = prefix + '2021-high-density/' + suffix1
#    dir1 = prefix + '2021-add/' + suffix1
#    dir1 = prefix + '2023-go/' + suffix1


def auto_divide(vaspruns):

    dft_dict = parse_vaspruns(vaspruns)

    # train0, test0 = [], []
    # train1, train2, test1, test2 = split_two_datasets(dft_dict)
    train0, train1, train2, test0, test1, test2 = split_three_datasets(dft_dict)

    print(" - Subset size (train1):", len(train1))
    print(" - Subset size (train2):", len(train2))
    print(" - Subset size (train_high_e):", len(train0))
    print(" - Subset size (test1): ", len(test1))
    print(" - Subset size (test2): ", len(test2))
    print(" - Subset size (test_high_e):", len(test0))

    vaspruns = np.array(vaspruns)
    f = open("polymlp.in.append", "w")
    print("", file=f)
    if len(train1):
        copy_vaspruns(vaspruns[train1], "train1")
        print("train_data vaspruns/train1/vaspruns-*.xml True 1.0")
        print("train_data vaspruns/train1/vaspruns-*.xml True 1.0", file=f)
    if len(train2):
        copy_vaspruns(vaspruns[train2], "train2")
        print("train_data vaspruns/train2/vaspruns-*.xml True 1.0")
        print("train_data vaspruns/train2/vaspruns-*.xml True 1.0", file=f)
    if len(train0):
        copy_vaspruns(vaspruns[train0], "train_high_e")
        print("train_data vaspruns/train_high_e/vaspruns-*.xml True 0.1")
        print(
            "train_data vaspruns/train_high_e/vaspruns-*.xml True 0.1",
            file=f,
        )

    if len(test1):
        copy_vaspruns(vaspruns[test1], "test1")
        print("test_data vaspruns/test1/vaspruns-*.xml True 1.0")
        print("test_data vaspruns/test1/vaspruns-*.xml True 1.0", file=f)
    if len(test2):
        copy_vaspruns(vaspruns[test2], "test2")
        print("test_data vaspruns/test2/vaspruns-*.xml True 1.0")
        print("test_data vaspruns/test2/vaspruns-*.xml True 1.0", file=f)
    if len(test0):
        copy_vaspruns(vaspruns[test0], "test_high_e")
        print("test_data vaspruns/test_high_e/vaspruns-*.xml True 0.1")
        print("test_data vaspruns/test_high_e/vaspruns-*.xml True 0.1", file=f)
    f.close()
