#!/usr/bin/env python
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))
