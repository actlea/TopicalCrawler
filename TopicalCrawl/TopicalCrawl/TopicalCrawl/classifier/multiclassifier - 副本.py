#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multiclass Naive Bayes SVM (NB-SVM)
https://github.com/lrei/nbsvm

Luis Rei <luis.rei@ijs.si>
@lmrei
http://luisrei.com

Learns a multiclass (OneVsRest) classifier based on word ngrams.
Uses scikit learn. Reads input from TSV files.

Licensed under a Creative Commons Attribution-NonCommercial 4.0
International License.

Based on a work at https://github.com/mesnilgr/nbsvm:
Naive Bayes SVM by Grégoire Mesnil
"""
from base import *
from sklearn.externals import joblib

import os
import numpy as np
import re

from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# from sklearn.svm.libsvm import predict_proba

from sklearn.linear_model import LogisticRegression


def en_tokenize(text):
    words = text.split()
    return [stem_words(word) for word in words if is_bad_tok(word)]


def zh_tokenize(text):
    words = term_seg(text)
    return [stem_words(word) for word in words if is_bad_tok(word)]


def normalize_text(text):
    if text is None or len(text)<2:
        return ''
    if not isinstance(text, unicode):
        text = text.decode('utf-8', 'ignore')

    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'[\r,\n]\s','',text) #去除行
    text =  u''.join([x.strip() for x in text.split() if len(x.strip())!=0])
    return text.encode('utf-8','ignore')


def convert_text(sentence, grams, custom_tokenize=zh_tokenize):
    # sentence = normalize_text(sentence)
    toks = custom_tokenize(sentence)
    ret = []
    for gram in grams:
        ret += ['_*_'.join(toks[i:i + gram]) for i in range(len(toks) - gram + 1)]
    return ret


def fill_config(dic, ratios, v, class2id, text_row=1, class_row=0, grams='12'):
    params = {
        'dic': dic,
        'ratios': ratios,
        'v': v,
        'text_row': text_row,
        'grams': grams,
        'class_row': class_row,
        'class2id': class2id
    }
    return params


# ======================================================================
# nbsvm function
def build_counters(filepath, grams, text_row, class_row, custom_tokenize=en_tokenize):
    """Reads text from a TSV file column creating an ngram count
    Args:
        filepath, the tsv filepath
        grams, the n grams to use
        text_row, row in the tsv file where the text is stored
        class_row, row in the tsv file where the class is stored
    """
    counters = {}
    class2id = {}
    text_src = read_text_src(filepath)
    for line in text_src:
        try:
            label, text = line[class_row], line[text_row]
        except ValueError or IndexError:
            print line
            print 'line error'
            continue
        if label not in class2id:
            class2id[label] = len(class2id)
        c = class2id[label]
        if c not in counters:
            counters[c] = Counter()
        counter = counters[c]
        counter.update(convert_text(text, grams, custom_tokenize))
    return counters, class2id


def compute_ratios(counters, alpha=1.0):
    """Computes the log-likelihood ratios for each class
    """
    ratios = dict()
    # create a vocabulary - a list of all ngrams
    all_ngrams = set()
    for counter in counters.values():
        all_ngrams.update(counter.keys())
    all_ngrams = list(all_ngrams)
    v = len(all_ngrams)  # the ngram vocabulary size
    print v

    # a standard NLP dictionay (ngram -> index map) use to update the
    # one-hot vector p
    dic = dict((t, i) for i, t in enumerate(all_ngrams))

    # for each class we create calculate a ratio (r_c)
    for c in counters.keys():
        p_c = np.full(v, alpha)
        counter = counters[c]

        for t in all_ngrams:
            p_c[dic[t]] += counter[t]

        # normalize (l1 norm)
        p_c /= np.linalg.norm(p_c, ord=1)
        ratios[c] = np.log(p_c / (1 - p_c))

    return dic, ratios, v


def count_lines(data_file):
    """Counts the number of lines in a file
    """
    with open(data_file) as f:
        lines = f.readlines()
        return len(lines)


def to_msvm(single_text, dic, v, ratios, grams, custom_tokenize, classes):
    X, data , indptr, indices= {}, {}, [0], []
    for c in classes:
        data[c] = []

    ngrams = convert_text(single_text, grams, custom_tokenize)
    for g in ngrams:
        if g in dic:
            index = dic[g]
            indices.append(index)
            for c in classes:
                data[c].append(ratios[c][index])
    indptr.append(len(indices))

    for c in classes:
        X[c] = csr_matrix((data[c], indices, indptr), shape=(1, v), dtype=np.float32)

    return X




def load_data(data_path, text_row, class_row, dic, v, ratios, grams, class2id, custom_tokenize=en_tokenize):
    """Create Train or Test matrix and Ground Truth Array
    """
    n_samples = count_lines(data_path)
    n_r = len(ratios)
    classes = ratios.keys()
    Y_real = np.zeros(n_samples, dtype=np.int64)

    # One X (sample) matrix and binary Y (truth) per class
    X, Y, data, indptr, indices = {}, {}, {}, [0], []

    for c in classes:
        Y[c] = np.zeros(n_samples, dtype=np.int64)
        data[c] = []
    text_src = read_text_src(data_path)
    n = 0
    for line in text_src:
        try:
            label, text = line[class_row], line[text_row]
        except ValueError:
            print 'line error'
            continue
        t = class2id[label]
        for c in classes:
            Y[c][n] = int(c == t)
        Y_real[n] = t
        ngrams = convert_text(text, grams, custom_tokenize)
        for g in ngrams:
            if g in dic:
                index = dic[g]
                indices.append(index)
                for c in classes:
                    # X[c][n][idx] = ratios[c][idx]
                    data[c].append(ratios[c][index])
        indptr.append(len(indices))
        n += 1

    for c in classes:
        X[c] = csr_matrix((data[c], indices, indptr), shape=(n_samples, v),
                          dtype=np.float32)

    return X, Y, Y_real





class MultiClassifier(object):
    def __init__(self, name, method=LogisticRegression,custom_tokenize=zh_tokenize):
        self.name = name
        self.method = method
        self.train_svm_file = '%s_train.svm' % name
        self.models = None
        self.params = None
        self.custom_tokenize = custom_tokenize

    def train(self, train_file, text_row=1, class_row=0, ngrams="12"):
        grams = [int(i) for i in ngrams]

        print '------------build counters----------------'
        counters, class2id = build_counters(train_file, grams, text_row, class_row, self.custom_tokenize)
        print '------------compute_ratios----------------'
        dic, ratios, v = compute_ratios(counters)
        classes = ratios.keys()
        print '-------------load data--------------------'
        Xs_train, Ys_train, y_train = load_data(train_file, text_row, class_row, dic, v, ratios, grams, class2id,
                                                self.custom_tokenize)

        print '-------------train classfier--------------------'
        svms = dict()
        for c in classes:
            svms[c] = self.method()
            # svms[c] = LogisticRegression()
            svms[c].fit(Xs_train[c], Ys_train[c])
        self.models = svms

        self.params = fill_config(dic, ratios, v, class2id, text_row, class_row, grams)

    def predict(self, single_text):
        class2id = self.params['class2id']
        classes = class2id.values()
        id2class = {_id:_class for _class, _id in class2id.items()}

        Xs_test = to_msvm(single_text, self.params['dic'], self.params['v'], self.params['ratios'],
                    self.params['grams'], self.custom_tokenize, classes)


        preds = dict()  #predic class
        probs = dict()
        for c in classes:
            preds[c] = self.models[c].decision_function(Xs_test[c])
            if hasattr(self.models[c], "predict_proba"):
                probs[c] = self.models[c].predict_proba(Xs_test[c])[:, 1]
            else:
                probs[c] = (preds[c]-preds[c].min())/(preds[c].max() - preds[c].min())

        max_score = float('-inf')
        pred,prob = classes[0], 0.5
        for c in classes:
                if preds[c][0] > max_score:
                    max_score = preds[c][0]
                    pred = c
                    prob = probs[c][0]

        return id2class[pred], prob



    def test(self, test_file):
        Xs_test, Ys_test, y_true = load_data(test_file, self.params['text_row'],
                                             self.params['class_row'], self.params['dic'],
                                             self.params['v'], self.params['ratios'],
                                             self.params['grams'], self.params['class2id'],
                                             self.custom_tokenize)
        classes = self.params['class2id'].values()
        preds = dict()  #predic class
        probs = dict() #predict probablity
        for c in classes:
            preds[c] = self.models[c].decision_function(Xs_test[c])
            if hasattr(self.models[c], "predict_proba"):
                probs[c] = self.models[c].predict_proba(Xs_test[c])[:, 1]
            else:
                probs[c] = (preds[c]-preds[c].min())/(preds[c].max() - preds[c].min())

        # not calculate the argmax
        pred = np.zeros(len(y_true))
        prob = np.zeros(len(y_true))
        for idx in range(0, len(y_true)):
            max_score = float('-inf')
            for c in classes:
                if preds[c][idx] > max_score:
                    max_score = preds[c][idx]
                    pred[idx] = c
                    prob[idx] = probs[c][idx]
        acc_svm = accuracy_score(y_true, pred)
        print('NBSVM: %f' % (acc_svm,))
        return TestResult(y_true, pred, prob)

    def save(self):
        if not os.path.exists(self.name):
            os.mkdir(self.name)
        joblib.dump(self.models, self.name + '/models.pkl')
        joblib.dump(self.params, self.name + '/params.pkl')

    def load(self, path=None):
        if path is None:
            self.models = joblib.load(self.name + '/models.pkl')
            self.params = joblib.load(self.name + '/params.pkl')
            return self
        else:
            home_path = os.path.join(path, self.name)
            self.models = joblib.load(home_path + '/models.pkl')
            self.params = joblib.load(home_path + '/params.pkl')
            return self





if __name__ == '__main__':
    d = './sample-data/train-jc.txt'
    t = './sample-data/test-en.csv'
    mc = MultiClassifier('test-jc', custom_tokenize=en_tokenize)
    # d = './sample-data/train-jc.txt'
    # t = './sample-data/test-jc.txt'
    # mc = MultiClassifier('test-jc-logreg', custom_tokenize=zh_tokenize)
    import time
    start = time.time()
    mc.train(d)
    mc.save()
    mc.load()
    # test_result = mc.test(t)
    # cost = time.time()-start
    # print 'cost time:', cost
    # print test_result.accuracy_labels
    # print test_result.recall_labels
    # print test_result.accuracy_score
    # test_result.show_result()
    print mc.predict('后场 篮板 被 拿到')
    print mc.predict('14英尺外 转身 后仰 打板 跳投 不中')
    print mc.predict('助攻 反身 扣篮 得2分')
    print mc.predict('26 英尺外 三分 跳投 不中')
    # print mc.predict('本报讯昨天，“新湖杯英才美女会”围棋对抗赛在杭州战罢首轮，结果五位头顶世界冠军头衔的须眉男儿，面对妩媚娇小的对手手下毫不留情，五战全胜的成绩让美女们颇有几分花容失色')
