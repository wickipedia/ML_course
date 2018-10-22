# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np


def compute_mse(y, tx, w):
	e = y - tx.dot(w)
	mse = e.dot(e) / (2 * len(e))
	return mse
