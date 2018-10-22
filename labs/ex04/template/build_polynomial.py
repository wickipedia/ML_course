# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, deg):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1),dtype='float64')
    for deg in range(1, deg+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
