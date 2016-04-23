#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: biclassifier.py
@time: 16-4-23 上午10:28
"""
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize, binarize, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from pure_url_classify import url_tokenize,url_tokenize2
from base import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report

tokenizes={'zh':zh_tokenize, 'en':en_tokenize, 'url':url_tokenize, 'url2':url_tokenize2}

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