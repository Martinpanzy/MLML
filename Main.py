#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:01:04 2018

@author: panzengyang
"""
import SVM_cv
import Perceptron_cv
import msg2matrix as mx
from sklearn.metrics import hinge_loss
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from msg2matrix import trainmsg, trainresult, testmsg, testresult
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
vectorizer = TfidfVectorizer(max_df = 0.20, min_df = 5, max_features = 1000, sublinear_tf = True)
vectorizer.fit(trainmsg)
dictionary = vectorizer.get_feature_names()

train_matrix = mx.extract_features(trainmsg,dictionary)
test_matrix = mx.extract_features(testmsg, dictionary)

per = Perceptron(max_iter = 5, eta0 = 1, class_weight = "balanced")
per.fit(train_matrix, trainresult)
decision = per.decision_function(test_matrix)
hloss = hinge_loss(testresult, decision)  
#Eout_per = 1 - per.score(test_matrix, testresult)

#clf = SVC(kernel='linear', C=0.1)
#clf.fit(train_matrix, trainresult)
#Eout_svm = 1 - clf.score(test_matrix, testresult)


#NB = MultinomialNB()
#NB.fit(train_matrix, trainresult)
#Eout_nb = 1 - NB.score(test_matrix, testresult)

#NBB = BernoulliNB()
#NBB.fit(train_matrix, trainresult)
#Eout_nbb = 1 - NBB.score(test_matrix, testresult)

#list_eta = [0.01, 1000000]
#per = Perceptron(max_iter = 3, eta0 = Perceptron_cv.percep_cv(list_eta, trainmsg, trainresult), class_weight = "balanced")
#per.fit(train_matrix, trainresult)
#Eout_per = 1 - per.score(test_matrix, testresult)

#list_C = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
#clf = SVC(kernel = 'linear', C = SVM_cv.svm_cv(list_C, trainmsg, trainresult))
#clf.fit(train_matrix, trainresult)
#Eout_svm = 1 - clf.score(test_matrix, testresult)