#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: cotraining.py
@time: 16-4-15 下午4:14
"""



from base import *
from pure_url_classify import url_tokenize,url_tokenize2

import time
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import cPickle
from sklearn.linear_model import LogisticRegression, SGDClassifier

import os
from sklearn.externals import joblib

tokenizes={'zh':zh_tokenize, 'en':en_tokenize, 'url':url_tokenize, 'url2':url_tokenize2}


def get_label(pred, prob, posLabel='0', threshold=0.7):
    if prob<threshold:
        return 0
    if pred==posLabel:
        return 1
    else:
        return -1


def sign(x):
    if x>=0:
        return '0'
    else:
        return '1'



def document_predict(textModel, urlModel, document, posLabel='0', W=[3,2,2,1]):
    title = document['title']
    headlines = document['headlines']
    base_url = document['base_url']
    meta = document['meta_descrip']
    para = document['para']

    url_preb, url_prob = urlModel.predict(base_url)

    if len(title)>2:
        t_preb, t_prob = textModel.predict(title)
    else:
        t_preb , t_prob = 0, 0

    if len(headlines)>2:
        h_preb, h_prob = textModel.predict(headlines)
    else:
        h_preb, h_prob = 0, 0

    if len(meta) > 2:
        m_preb, m_prob = textModel.predict(meta)
    else:
        m_preb, m_prob = 0, 0

    if len(para) > 2:
        p_preb, p_prob = textModel.predict(para)
    else:
        p_preb, p_prob = 0, 0

    t_preb = get_label(t_preb, t_prob,posLabel)
    h_preb = get_label(h_preb, h_prob,posLabel)
    m_preb = get_label(m_preb, m_prob, posLabel)
    p_preb = get_label(p_preb, p_prob, posLabel)

    text_pred = W[0]*t_preb+W[1]*h_preb+W[2]*m_preb+W[3]*p_preb
    text_preb = sign(text_pred)

    #返回与预测结果相同的最大的概率值作为text预测的概率
    max_prob = 0
    if text_preb==posLabel:
        tmp = 1
    else:
        tmp = -1
    for i in [(t_preb,t_prob), (h_preb,h_prob), (m_preb,m_prob), (p_preb, p_prob)]:
        if i[0]==tmp and i[1]>max_prob:
            max_prob = i[1]

    # return url_preb, url_prob, text_preb, max_prob
    if url_preb==text_preb:
        return url_preb, max(max_prob, url_prob)
    elif max_prob<url_prob:
        return url_preb, url_prob
    else:
        return text_preb,max_prob






class MultiClassifier(object):
    def __init__(self, name, method=LogisticRegression, tokenize='zh', grams=[1]):
        tokenize = tokenizes[tokenize]
        self.name = name
        self.method = method
        self.models = {}
        self.converter = Converter(tokenize, grams)


    def train(self, X_train, y_train):
        X, Y = self.converter.fit_transform(X_train, y_train)
        models = {}
        classes = self.converter.get_classes()
        for c in classes:
            models[c]=self.method()
            models[c].fit(X[c], Y[c])
        self.models = models

    def predicts(self, X_test):
        '''
        :param X_test: list of text
        :return: list of predict labels
        '''
        X = self.converter.transform(X_test)
        preds = {}
        probs = {}
        for c in self.converter.get_classes():
            preds[c] = self.models[c].decision_function(X[c])
            if hasattr(self.models[c], "predict_proba"):
                probs[c] = self.models[c].predict_proba(X[c])[:, 1]
            else:
                probs[c] = (preds[c] - preds[c].min()) / (preds[c].max() - preds[c].min())

        pred_result = np.zeros(len(X_test))
        prob_result = np.zeros(len(X_test))
        for idx in range(0, len(X_test)):
            max_score = float('-inf')
            for c in self.converter.get_classes():
                if preds[c][idx] > max_score:
                    max_score = preds[c][idx]
                    pred_result[idx] = c
                    prob_result[idx] = probs[c][idx]
        return pred_result, prob_result


    def predict(self, single_text):
        pred, prob = self.predicts([single_text])
        return pred[0], prob[0]


    def test(self, X_train, y_train):
        pred_result, prob_result = self.predicts(X_train)
        Y_real = np.zeros(len(X_train), dtype=np.int64)
        for i in range(len(y_train)):
            Y_real[i] = int(y_train[i])

        print classification_report(Y_real, pred_result)
        print 'accuracy score: %s' %accuracy_score ( Y_real, pred_result)

    def save(self):
        if not os.path.exists(self.name):
            os.mkdir(self.name)
        self.converter.save(self.name)
        joblib.dump(self.models, self.name + '/models.pkl')

    def load(self, path=None):
        if path is not None:
            dest_src = os.path.join(path, self.name)
        else:
            dest_src = self.name

        self.converter.load(dest_src)
        self.models = joblib.load(dest_src + '/models.pkl')
        return self


def create_train_test(d, t, classifier):
    lines = read_text_src(d)
    lines = [x for x in lines if len(x) > 1]
    X_train = [line[1] for line in lines]
    y_train = [line[0] for line in lines]

    lines = read_text_src(t)
    lines = [x for x in lines if len(x) > 1]
    X_test = [line[1] for line in lines]
    y_test = [line[0] for line in lines]

    s = time.time()
    classifier.train(X_train, y_train)
    classifier.save()
    classifier.test(X_test, y_test)
    t=time.time()-s
    print 'cost time:', t

def test(t, classifier):
    lines = read_text_src(t)
    lines = [x for x in lines if len(x) > 1]
    X_test = [line[1] for line in lines]
    y_test = [line[0] for line in lines]

    s = time.time()
    classifier.load()
    classifier.test(X_test, y_test)
    t = time.time() - s
    print 'cost time:', t

def train(d, classifier):
    lines = read_text_src(d)
    lines = [x for x in lines if len(x) > 1]
    X_train = [line[1] for line in lines]
    y_train = [line[0] for line in lines]
    s = time.time()
    classifier.train(X_train, y_train)
    classifier.save()
    t = time.time() - s
    print 'cost time:', t


def load_classifier(path):

    url_1_model = MultiClassifier('test-3M_-topic', method=SGDClassifier, tokenize='url2')
    url_1_model.load(path)

    anchor_1_model = MultiClassifier('test-2M-topic' , tokenize='zh', grams=[1,2])
    anchor_1_model.load(path)

    para_1_model = MultiClassifier('test-1M-topic', tokenize='zh', grams=[1, 2])
    para_1_model.load(path)

    anchor_2_model = MultiClassifier('test-5M-topic', tokenize='zh', grams=[1,2])
    anchor_2_model.load(path)

    para_2_model = MultiClassifier('test-6M-topic', tokenize='zh', grams=[1, 2])
    para_2_model.load(path)

    return url_1_model, anchor_1_model, para_1_model, anchor_2_model, para_2_model


if __name__=='__main__':
    # path = '/mnt/UbutunShare/graduate/DataSet/document1.txt'
    number = '1'
    d = './sample-data/train-%s-zh.txt' % number
    t = './sample-data/test-%s-zh.txt' % number


    c1 = MultiClassifier('test-1M-topic', tokenize='zh', grams=[1, 2])
    # create_train_test(d, t, c1)

    # 短文本分类器===============================================================
    number = 2
    d = './sample-data/train-%s-zh.txt' % number
    t = './sample-data/test-%s-zh.txt' % number
    c2 = MultiClassifier('test-2M-topic' , tokenize='zh', grams=[1,2])
    # create_train_test(d, t, c2)
    # train(d, c2)
    # test(t, c1)

    # url分类器===============================================================
    c3 = MultiClassifier('test-3M_-topic', method=SGDClassifier, tokenize='url2')
    number = 3
    d = './sample-data/train-%s-zh.txt' % number
    t = './sample-data/test-%s-zh.txt' % number
    # create_train_test(d, t, c3)


    #second level classifier
    #=======================================================================
    c5 = MultiClassifier('test-5M-topic', tokenize='zh', grams=[1,2])
    number = 5
    d = './sample-data/train-%s-zh.txt' % number
    t = './sample-data/test-%s-zh.txt' % number
    create_train_test(d, t, c5)






    #=====================================================================
    #url
    # c5 = MultiClassifier('test-3M_-topic', method=SGDClassifier, tokenize='url2')
    # number = 5
    # d = './sample-data/train-%s-zh.txt' % number
    # t = './sample-data/test-%s-zh.txt' % number


















