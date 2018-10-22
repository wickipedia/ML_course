#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:16:09 2018

@author: elias
"""

# Useful starting lines
import numpy as np
from cross_validation import *
from helpers import load_data

# load dataset
x, y = load_data()

#%%

cross_validation_demo(x,y)

        

    

