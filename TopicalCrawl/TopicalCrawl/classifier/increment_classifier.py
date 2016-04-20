#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: increment_classifier.py
@time: 16-4-15 上午10:18
"""
from base import *
from sklearn.externals import joblib
import os
import numpy as np
import re
from collections import  Counter

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def get_bn_ratios(X_train, y_train, v):
    print v
    matrix = X_train.toarray()
    class_id = set(y_train)
    ratios = np.ones((len(class_id), v))
    counters = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print matrix[i][j],
        print ''

    # for c in class_id:
    #     p_c = np.full(v, 1.0)
    #     for t in range(v):
    #         p_c[t] += counters[t]
    #         # normalize (l1 norm)
    #     p_c /= np.linalg.norm(p_c, ord=1)
    #     ratios[c] = np.log(p_c / (1 - p_c))
    # print ratios, v







class IncrementalClassifier(object):

    def __init__(self, name):
        self.name = name
        self.model = None
        self.converter = TextPreProcess(zh_tokenize)


    def train_and_test(self, train_file, test_file):
        lines = read_text_src(train_file)
        lines = [x for x in lines if len(x) > 1]
        X_train = [line[1] for line in lines]
        y_train = [line[0] for line in lines]

        # lines = read_text_src(test_file)
        # lines = [x for x in lines if len(x) > 1]
        # X_test = [line[1] for line in lines]
        # y_test = [line[0] for line in lines]

        vectorizer = CountVectorizer(tokenizer=zh_tokenize)  # ngram_range=(1,2)

        X_train = vectorizer.fit_transform(X_train)
        print type(X_train)
        # X_test = vectorizer.transform(X_test)
        word = vectorizer.get_feature_names()
        v = len(word)
        get_bn_ratios(X_train,y_train,v)

        N = X_train.shape[1]
        ch2 = SelectKBest(chi2, k=int(N * 0.2))
        X_train = ch2.fit_transform(X_train, y_train)
        feature_names = [word[i] for i
                         in ch2.get_support(indices=True)]


if __name__=='__main__':
    path = '/mnt/UbutunShare/graduate/DataSet/document1.txt'
    test_file = 'sample-data/test-6-zh.txt'
    inc = IncrementalClassifier('test')
    inc.train_and_test(path,test_file)
