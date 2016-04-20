#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: test.py
@time: 16-4-16 下午10:59
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from classifiers import CoTrainingClassifier

'''
if __name__ == '__main__':
    N_SAMPLES = 25000
    N_FEATURES = 1000
    X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)

    y[:N_SAMPLES//2] = -1

    X_test = X[-N_SAMPLES//4:]
    y_test = y[-N_SAMPLES//4:]

    X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
    y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

    y = y[:-N_SAMPLES//4]
    X = X[:-N_SAMPLES//4]


    X1 = X[:,:N_FEATURES // 2]
    X2 = X[:, N_FEATURES // 2:]




    print 'Logistic'
    base_lr = LogisticRegression()
    base_lr.fit(X_labeled, y_labeled)
    y_pred = base_lr.predict(X_test)
    print classification_report(y_test, y_pred)

    print 'Logistic CoTraining'
    lg_co_clf = CoTrainingClassifier(LogisticRegression())
    lg_co_clf.fit(X1, X2, y)
    y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print classification_report(y_test, y_pred)

    from TopicalCrawl.classifier.multiclassifier import BiClassifier

    print 'BiClassifier '
    base_lr = BiClassifier('test3', feature_select=False, need_converter=False)
    base_lr.train(X_labeled, y_labeled)
    y_pred = base_lr.predict(X_test)
    print classification_report(y_test, y_pred)



    # print 'SVM'
    # base_svm = LinearSVC()
    # base_svm.fit(X_labeled, y_labeled)
    # y_pred = base_lr.predict(X_test)
    # print classification_report(y_test, y_pred)
    #
    # print 'SVM CoTraining'
    # svm_co_clf = CoTrainingClassifier(LinearSVC(), u=N_SAMPLES//10)
    # svm_co_clf.fit(X1, X2, y)
    # y_pred = svm_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    # print classification_report(y_test, y_pred)

    '''

from TopicalCrawl.classifier.multiclassifier import *
from TopicalCrawl.classifier.pure_url_classify import *

def sample(d, tokenize,grams=None):
    lines = read_text_src(d)
    lines = [x for x in lines if len(x) > 1]
    X_train = [line[1] for line in lines]
    y_train = [line[0] for line in lines]
    if grams is not None:
        vet = CountVectorizer(tokenizer=tokenize, ngram_range=grams)
    else:
        vet = CountVectorizer(tokenizer=tokenize)
    X_train = vet.fit_transform(X_train)

    return X_train, y_train

if __name__=='__main__':
    url_label='../../sample-data/train-9-zh.txt'
    page_label='../../sample-data/train-1-zh.txt'

    url_unlabel='../../sample-data/test-9-zh.txt'
    page_unlabel='../../sample-data/test-1-zh.txt'



    XL_url,YL_url = sample(url_label, url_tokenize, (1,3))
    XL_page, YL_page = sample(page_label, zh_tokenize)

    XUL_url, YUL_url = sample(url_unlabel, url_tokenize, (1,3))
    XUL_page, YUL_page = sample(page_unlabel, zh_tokenize)
    print 'Logistic'

    base_lr = LogisticRegression()
    base_lr.fit(XL_url, YL_url)
    y_pred = base_lr.predict(X_test)
    print classification_report(y_test, y_pred)

    print 'Logistic CoTraining'
    lg_co_clf = CoTrainingClassifier(LogisticRegression())
    lg_co_clf.fit(X1, X2, y)
    y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
    print classification_report(y_test, y_pred)



