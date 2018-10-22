# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    '''
    Least squares regression using normal equations
    arguments:
        y - output data, target
        tx - input data, features
    return:
        w - weigth of last iteration
        loss - mean square error of last iteration
    '''
    #return np.linalg.lstsq(np.transpose(tx).dot(tx), np.transpose(tx).dot(y),rcond=None)[0]
    w=np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    return w