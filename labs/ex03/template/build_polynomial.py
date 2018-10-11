# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):

    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE)],degree)
    Phi=np.zeros((x.size,degree+1))
    
    for i in range(degree):
        Phi[:,i]=x**i
    return Phi
