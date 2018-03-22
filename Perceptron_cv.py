#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:51:14 2018

@author: panzengyang
"""
import numpy as np
import msg2matrix as mx
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

def CrossValid_percep(eta, X, Y): #trainmsg, trainresult ALL
    Rcv = 0
    numsplit = 2
    kf = KFold(n_splits = numsplit)
    for train_index, test_index in kf.split(X):
        train_result, val_result = Y[train_index], Y[test_index]
        train_msg, val_msg = X[train_index], X[test_index]
        
        vectorizer = TfidfVectorizer(max_df = 0.35, min_df = 5, max_features = 500, sublinear_tf = True)
        vectorizer.fit(train_msg)
        dictionary = vectorizer.get_feature_names()
        
        train_matrix = mx.extract_features(train_msg, dictionary)
        val_matrix = mx.extract_features(val_msg, dictionary)
        
        per = Perceptron(max_iter = 1, eta0 = eta, class_weight = "balanced")
        per.fit(train_matrix, train_result)
        
        Rcv = Rcv + (1 - per.score(val_matrix, val_result))
    Rcv = Rcv/numsplit
    print("eta = ", eta, "    Rvc = ", Rcv)
    return Rcv
    

def percep_cv(list_eta, X, Y): #trainmsg, trainresult ALL
    Rcv = []
    for i, eta in enumerate(list_eta):
        Rcv.append(CrossValid_percep(eta, X, Y))
    index_min = np.argmin(Rcv)
    eta = list_eta[index_min]
    print("Correct eta = ", eta)
    
    return eta