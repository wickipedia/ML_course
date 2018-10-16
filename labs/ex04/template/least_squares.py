# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    return np.linalg.lstsq(np.transpose(tx).dot(tx), np.transpose(tx).dot(y),rcond=None)[0]
    #return np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))