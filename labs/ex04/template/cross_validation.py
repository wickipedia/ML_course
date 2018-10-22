#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:20:16 2018

@author: elias
"""
from costs import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly
import numpy as np
from plots import cross_validation_visualization




def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    loss_tr=[]
    loss_te=[]
    for j in range(k):
        
        index_te=k_indices[j]
        
        ind=np.ones(k_indices.shape[0],bool)
        ind[j]=False
        index_tr=k_indices[ind].flatten()
        
        x_tr=x[index_tr]
        x_te=x[index_te]
        y_tr=y[index_tr]
        y_te=y[index_te]
        
        xpoly_tr=build_poly(x_tr,degree)
        xpoly_te=build_poly(x_te,degree)   
        
        w_s=ridge_regression(y_tr,xpoly_tr,lambda_)
             
        loss_tr.append(compute_mse(y_tr,xpoly_tr,w_s))
        loss_te.append(compute_mse(y_te,xpoly_te,w_s))
        
    return np.mean(loss_tr), np.mean(loss_te)


def cross_validation_demo(x,y):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # *************************************************** 
    for l in lambdas:

        
        k_loss_tr, k_loss_te = cross_validation(y,x,k_indices,k_fold,l,degree)
            
        mean_loss_tr=(k_loss_tr)
        mean_loss_te=(k_loss_te)
        
        rmse_tr=np.append(rmse_tr,np.sqrt(mean_loss_tr))
        rmse_te=np.append(rmse_te,np.sqrt(mean_loss_te))
        
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
