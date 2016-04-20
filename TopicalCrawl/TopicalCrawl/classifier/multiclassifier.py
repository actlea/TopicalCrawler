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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
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


from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer



#CountVectorizer, TfidfVectorizer
class BiClassifier(object):
    def __init__(self, name, method=LogisticRegression,
                 need_converter=True,feature_select=False,
                 tokenize='zh', converter=TfidfVectorizer,grams=None):
        self.name = name
        self.feature_select = feature_select
        self.need_converter = need_converter
        self.method = method
        if not os.path.exists(self.name):
            self.model = None
            if grams is not None:
                self.converter = converter(tokenizer=tokenizes[tokenize], ngram_range=grams)
            else:
                self.converter = converter(tokenizer=tokenizes[tokenize])
            self.ch2 = None
        else:
            self.load()



    # def nbsvm_ratios(self, X_train, y_train, alpha=1.0):
    #     self.ratios_ = np.full((2, X_train.shape[1]), alpha,
    #                            dtype=np.float64)
    #     self.ratios_ += safe_sparse_dot(y_train.T, X_train)
    #     normalize(self.ratios_, norm='l1', axis=1, copy=False)
    #     row_calc = lambda r: np.log(np.divide(r, (1 - r)))
    #     self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
    #     self.ratios_ = csr_matrix(self.ratios_)
    #     if self.need_nbsvm:
    #         X_train = X_train.multiply(self.ratios_[0])
    #     # counter={}
    #     # for i in range(len(y_train)):
    #     #     counter.setdefault(y_train[i], []).append(X_train[i])
    #     #
    #     #
    #     # p_0 = np.full(len(y_train), alpha)
    #     # p_1 = np.full(len(y_train), alpha)
    #     # p_c = [p_0,p_1]
    #     # for i, samples in enumerate(counter):
    #     #     for samp in samples:
    #     #         p_c[i] += samp
    #     #
    #     # p_0 /= np.linalg.norm(p_c[0], ord=1)
    #     # p_1 /= np.linalg.norm(p_c[1], ord=1)
    #     # ratios = np.log(p_0/p_1)




    def train(self, X_train, y_train, ratio=0.4):
        if self.need_converter:
            X_train = self.converter.fit_transform(X_train)
        if self.feature_select:
            N = X_train.shape[1]
            self.ch2 = SelectKBest(chi2, k=int(N * ratio))
            X_train = self.ch2.fit_transform(X_train, y_train)

        clf = self.method()
        clf.fit(X_train, y_train)
        self.model = clf

    def save(self):
        if not os.path.exists(self.name):
            os.mkdir(self.name)

        if self.feature_select:
            dic_ch2 = {'ch2': self.ch2}
            cPickle.dump(dic_ch2, open(self.name + '/dic_ch2.pkl', 'wb'), -1)

        joblib.dump(self.model, self.name + '/model.pkl')
        if self.need_converter:
            dic_con = {'con': self.converter}
            cPickle.dump(dic_con, open(self.name + '/dic_con.pkl', 'wb'), -1)

    def test(self,X_test, y_test):
        if self.need_converter:
            X_test = self.converter.transform(X_test)
        if self.feature_select:
            X_test = self.ch2.transform(X_test)
        pred = self.model.predict(X_test)
        print classification_report(y_test, pred)

    @staticmethod
    def show_result(pred, y_true):
        print classification_report(y_true, pred)


    def load(self, path=None):
        if path is not None:
            dest_src = os.path.join(path, self.name)
        else:
            dest_src = self.name

        self.model = joblib.load(dest_src + '/model.pkl')
        if self.need_converter:
            self.converter = cPickle.load(open(dest_src+'/dic_con.pkl', 'rb'))['con']
        if self.feature_select:
            self.ch2 = cPickle.load(open(dest_src+'/dic_ch2.pkl', 'rb'))['ch2']

        return self

    def predict(self, text):
        if self.need_converter:
            X_train=[text]
            X_train = self.converter.transform(X_train)
        if self.feature_select:
            X_train = self.ch2.transform(X_train)
        pred = self.model.predict(X_train)
        prob = self.model.predict_proba(X_train)
        if pred=='0':
            return pred[0],prob[0][0]
        else:
            return pred[0],prob[0][1]

    def predicts(self, X_test):
        if self.need_converter:
            X_test = self.converter.transform(X_test)
        if self.feature_select:
            X_test = self.ch2.transform(X_test)
        pred = self.model.predict(X_test)
        # prob = self.model.predict_proba(X_test)
        #
        # pred_r=[]
        # prob_r = []
        # for i in range(len(pred)):
        #     pred_r.append(pred[0])
        #     prob_r.append(max(prob[0]))
        # return pred_r, prob_r
        return pred




def create_train_test(d, t, classifier):
    lines = read_text_src(d)
    lines = [x for x in lines if len(x) > 1]
    X_train = [line[1] for line in lines]
    y_train = [line[0] for line in lines]

    lines = read_text_src(t)
    lines = [x for x in lines if len(x) > 1]
    X_test = [line[1] for line in lines]
    y_test = [line[0] for line in lines]


    classifier.train(X_train, y_train)
    classifier.save()
    classifier.test(X_test, y_test)






if __name__=='__main__':
    # path = '/mnt/UbutunShare/graduate/DataSet/document1.txt'
    number = '1'
    d = './sample-data/train-%s-zh.txt' % number
    t = './sample-data/test-%s-zh.txt' % number

    c0 = BiClassifier('test-3-topic', feature_select=False, tokenize='zh', grams=(1,2))
    # create_train_test(d, t, c0)



    c1 = MultiClassifier('test-3-text' , tokenize='zh', grams=[1,2])
    create_train_test(d, t, c1)
    # t = './sample-data/test-%s-zh.txt' % '1'
    # lines = read_text_src(t)
    # lines = [x for x in lines if len(x) > 1]
    # X_train = [line[1] for line in lines]
    # y_train = [line[0] for line in lines]
    # # create_train_test(d, t, c1)
    # c1.load()
    # print c1.test(X_train, y_train)
    #==============================================================
    #test-9-text
    number = '9'
    d = './sample-data/train-%s-zh.txt' % number
    t = './sample-data/test-%s-zh.txt' % number


    #
    # c1 = BiClassifier('test-9-url-text', feature_select=False, tokenize='url',converter=CountVectorizer, grams=(1,3))
    # create_train_test(d,t,c1)
    # # c1.train(X_train, y_train)
    # # c1.save()
    # c1.test(X_test, y_test)
    # # c1.load()
    # print c1.predict(['http:sports.com/'])

    # c1 = MultiClassifier('test-9-url' , tokenize='url')
    # create_train_test(d, t, c1)
    # start = time.time()
    # c1.train(X_train, y_train)
    # c1.save()
    # # c1.load()
    # c1.test(X_test, y_test)
    # end = time.time()
    # print 'time cost: %s' % (end - start)
    # X_train=['库里神准三分','汤神强悍防守','今天股市大涨','库里','库里']
    # Y_train=['0','0','1','0','0']
    # c1 = MultiClassifier('test', tokenize='zh')
    #
    # c1.train(X_train,Y_train)
    # print c1.predict('库里防守')
    # print c1.predict('股市')













