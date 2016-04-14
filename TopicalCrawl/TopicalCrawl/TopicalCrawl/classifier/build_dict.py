#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: build_dict.py
@time: 16-4-13 上午11:38
"""

from multiclassifier import zh_tokenize
from base import read_text_src
import json
from math import log
import cPickle
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def dic_add(dic, key, cnt=1):
    if key in dic:
        dic[key] += cnt
    else:
        dic[key] = cnt

def unicode2str(text):
    if isinstance(text, str):
        return text
    else:
        return text.encode('utf-8')


def is_rarity_ok(Rmax, Rth, k, LDC, TFIDF, Tth):
    if LDC<Rmax and TFIDF>Rth:
        return True
    for i in range(1,k+1):
        if 2**(i-1)*Rmax<LDC<2**(i)*Rmax and TFIDF>Tth:
            return True
    return False

def is_popularity_ok(Tth, Pmax, LDC, TFIDF):
    return TFIDF>Tth and LDC<Pmax


def compute_tfidf(path, dest_dir, N=150):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    with open(path) as fr:
        lines = fr.read()
        lines = lines.splitlines()
        lines = [line.split('\t')[1] for line in lines]


    tok2tf = {}
    tok2idf = {}
    tok2tfidf = {}
    bgram2LDC = {}
    tok2LDC = {}

    popularity = {}
    rarity = {}

    for index, line in enumerate(lines):
        toks = zh_tokenize(line)

        #收集bgram词汇，统计两个词同时出现的频率
        gram = 2
        for i in range(len(toks) - gram + 1):
            bgram=tuple(toks[i:i + gram])
            if bgram not in bgram2LDC:
                bgram2LDC[bgram]=1

            # toks.append(tuple(toks[i:i + gram]))
        # 计算tf值
        for tok in toks:
            dic_add(tok2tf, tok)
            tok2idf.setdefault(tok, set()).add(index)
    N = len(lines)
    #compute tf*idf
    for tok in tok2tf:
        Nt = len(tok2idf[tok])
        tf = 1.0*tok2tf[tok]
        idf = log(N/Nt+0.01)
        weight = tf*idf
        tok2tfidf[tok] = weight
        tok2idf[tok] = idf

    # 计算联合出现的频率
    for bgram, cnt in bgram2LDC.items():
        left = bgram[0]
        right = bgram[1]
        dic_add(tok2LDC, left)
        dic_add(tok2LDC, right)
    for tok in tok2tf:
        if tok not in tok2LDC:
            tok2LDC[tok] = 1

    #选择出现频率最高的前20%的词作为popularity
    tfidf_tmp = sorted(tok2tfidf.items(), key=lambda x:x[1], reverse=True)
    LDC_tmp = sorted(tok2LDC.items(), key=lambda x: x[1], reverse=True)

    ratio = N*1.0/len(tfidf_tmp)
    #选择尽可能不那么普遍的词
    Tth = tfidf_tmp[int(len(tfidf_tmp)*ratio)][1]
    Pmax = LDC_tmp[int(len(LDC_tmp)*0.1)][1]
    popularity = {tok: freq for tok, freq in tok2tfidf.items() if is_popularity_ok(Tth, Pmax, tok2LDC[tok], freq)}

    Rth = tfidf_tmp[int(len(tfidf_tmp)*(1-ratio))][1] #在选择rarity时要求词频不得太小
    Rmax = LDC_tmp[int(len(LDC_tmp)*(1-ratio))][1]
    rarity = {tok: freq for tok, freq in tok2tfidf.items() if is_rarity_ok(Tth, Pmax, 10, tok2LDC[tok], freq, Tth)}





    #求联合出现最多的词项
    maxLDC = max(tok2LDC.values())
    print 'maxLDC:', maxLDC

    cPickle.dump(tok2tf, open('%s/tok2tf.pkl' %dest_dir, 'wb'), -1)
    cPickle.dump(tok2idf, open('%s/tok2idf.pkl' %dest_dir, 'wb'), -1)
    cPickle.dump(bgram2LDC, open('%s/bgram2LDC.pkl' %dest_dir, 'wb'), -1)

    with open('%s/tok2tf.txt' %dest_dir, 'w') as fw:
        for tok, tf in tok2tf.items():
            fw.write('%s\t%d\n' % (unicode2str(tok), tf))

    with open('%s/popularity.txt' % dest_dir, 'w') as fw:
        for tok, tfidf in popularity.items():
            fw.write('%s\t%d\n' % (unicode2str(tok), tfidf))

    with open('%s/rarity.txt' % dest_dir, 'w') as fw:
        for tok, tfidf in rarity.items():
            fw.write('%s\t%d\n' % (unicode2str(tok), tfidf))


    cPickle.dump(tok2tfidf, open('%s/tok2tfidf.pkl' %dest_dir, 'wb'), -1)
    with open('%s/tok2tfidf.txt' %dest_dir, 'w') as fw:
        for tok, tfidf in tok2tfidf.items():
            fw.write('%s\t%s\n' % (unicode2str(tok), str(tfidf)))

    #compute LDC
    cPickle.dump(tok2LDC, open('%s/tok2LDC.pkl' %dest_dir, 'wb'), -1)
    with open('%s/tok2LDC.txt' %dest_dir, 'w') as fw:
        for tok, LDC in tok2LDC.items():
            fw.write('%s\t%s\n' % (unicode2str(tok), str(LDC)))

from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def chi_feature_select(train_file, test_file):

    lines = read_text_src(train_file)
    lines = [x for x in lines if len(x)>1]
    X_train = [line[1] for line in lines]
    y_train = [line[0] for line in lines]

    lines = read_text_src(test_file)
    lines = [x for x in lines if len(x) > 1]
    X_test = [line[1] for line in lines]
    y_test = [line[0] for line in lines]

    vectorizer = TfidfVectorizer(tokenizer=zh_tokenize)#ngram_range=(1,2)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    word = vectorizer.get_feature_names()

    N = X_train.shape[1]
    ch2 = SelectKBest(chi2, k=int(N*0.2)) #.fit_transform(X, y)


    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)

    feature_names = [word[i] for i
                     in ch2.get_support(indices=True)]
    #

    # for i in feature_names:
    #     print i.encode('utf-8')
    # feature_names = np.asarray(feature_names)
    # print feature_names
    clf = LinearSVC(penalty="l1", dual=False, tol=1e-3)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)


import lda
def test_lda(train_file, out_file):
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
    # X_test = vectorizer.transform(X_test)
    vocab = vectorizer.get_feature_names()

    model = lda.LDA(n_topics=1, random_state=0, n_iter=1000)
    model.fit_transform(X_train, y_train)
    topic_word = model.components_  # model.components_ also works
    N = len(vocab)
    n_top_words = N*0.2
    with open(out_file, 'w') as fw:
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            print '%d:%s' %(i, '\n'.join(topic_words))
            data = '\n'.join(topic_words)
            data = data.encode('utf-8')
            fw.write('%s' %'\n'.join(topic_words))




if __name__=='__main__':
    path = '/mnt/UbutunShare/graduate/DataSet/document1.txt'
    # path = '/mnt/UbutunShare/graduate/DataSet/Big/C000008.txt'
    # path = '/mnt/UbutunShare/graduate/DataSet/Big/C000014.txt'

    path = '/mnt/UbutunShare/graduate/DataSet/Dic/tennis_news.txt'
    train_file = 'sample-data/train-6-zh.txt'
    test_file = 'sample-data/test-6-zh.txt'
    # compute_tfidf(path, 'tmp')
    # chi_feature_select(train_file, test_file)
    test_lda(path, 'tennis.txt')
    # test()
    # test_chi()