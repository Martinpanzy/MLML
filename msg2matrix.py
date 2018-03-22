#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:51:39 2018

@author: panzengyang
"""

import os
import numpy as np

def loadSMSData(trainfile):
    f = open(os.path.expanduser(trainfile))
    smsWords = []
    result = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            result.append(-1)
        elif linedatas[0] == 'spam':
            result.append(1)
            
        words = linedatas[1]
        smsWords.append(words)
    return smsWords, result

def TextParse(text):
    import re
    regEx = re.compile(r'[^a-zA-Z0-9_]')
    words = regEx.split(text)
    return [word.lower() for word in words if len(word) > 0]

def extract_features(msg, dictionary): 
    features_matrix = np.zeros((len(msg), len(dictionary)))
    for lineID in range(len(msg)):
        words = TextParse(msg[lineID])
        for word in words:
            for i,d in enumerate(dictionary):
                if d == word:
                    features_matrix[lineID, i] = words.count(word)   
    return features_matrix


smsmsg = '/Users/panzengyang/Desktop/SMSSPAM/SMSSpamCollection'
allWords, allresult = loadSMSData(smsmsg)
allresult = np.array(allresult)
allWords = np.array(allWords)

testmsg = allWords[4459:5574]
testresult = allresult[4459:5574]

trainmsg = allWords[:4459]
trainresult = allresult[:4459]