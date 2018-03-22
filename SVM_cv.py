#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:20:54 2018

@author: panzengyang
"""
import numpy as np
import msg2matrix as mx
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

def CrossValid_svm(c, X, Y): #trainmsg, trainresult ALL
    Rcv = 0
    numsplit = 5
    kf = KFold(n_splits = numsplit)
    for train_index, test_index in kf.split(X):
        train_result, val_result = Y[train_index], Y[test_index]
        train_msg, val_msg = X[train_index], X[test_index]
        
        vectorizer = TfidfVectorizer(max_df = 0.35, min_df = 5, max_features = 1000, sublinear_tf = True)
        vectorizer.fit(train_msg)
        dictionary = vectorizer.get_feature_names()
        
        train_matrix = mx.extract_features(train_msg, dictionary)
        val_matrix = mx.extract_features(val_msg, dictionary)
        
        clf = SVC(kernel = 'linear', C = c)
        clf.fit(train_matrix, train_result)
        Rcv = Rcv + (1 - clf.score(val_matrix, val_result))
    Rcv = Rcv/numsplit
    print("C = ", c, "    Rvc = ", Rcv)
    return Rcv

def svm_cv(list_C, X, Y): #trainmsg, trainresult ALL
    Rcv = []
    for i, c in enumerate(list_C):
        Rcv.append(CrossValid_svm(c, X, Y))
    index_min = np.argmin(Rcv)
    c = list_C[index_min]
    print("Correct C = ", c)
    return c