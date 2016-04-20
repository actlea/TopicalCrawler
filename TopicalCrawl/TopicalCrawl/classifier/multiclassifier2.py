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
# from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def fill_config(dic, ratios, v, class2id, text_row=1, class_row=0, grams='12'):
    params={
        'dic': dic,
        'ratios': ratios,
        'v': v,
        'class2id': class2id
    }
    return params


# ======================================================================
# nbsvm function
def build_counters(filepath, grams, text_row, class_row, converter):
    """Reads text from a TSV file column creating an ngram count
    Args:
        filepath, the tsv filepath
        grams, the n grams to use
        text_row, row in the tsv file where the text is stored
        class_row, row in the tsv file where the class is stored
    """
    counters={}
    class2id={}
    text_src=read_text_src ( filepath )
    for line in text_src:
        if len(line)<text_row+1:continue
        try:
            label, text=line[ class_row ], line[ text_row ]
        except ValueError or IndexError:
            print line
            print 'line error'
            continue
        if label not in class2id:
            class2id[ label ]=len ( class2id )
        c=class2id[ label ]
        if c not in counters:
            counters[ c ]=Counter ( )
        counter=counters[ c ]
        counter.update ( converter.convert_text ( text, grams ) )
    return counters, class2id


def compute_ratios(counters, alpha=1.0):
    """Computes the log-likelihood ratios for each class
    """
    ratios=dict ( )
    # create a vocabulary - a list of all ngrams
    all_ngrams=set ( )
    for counter in counters.values ( ):
        all_ngrams.update ( counter.keys ( ) )
    all_ngrams=list ( all_ngrams )
    v=len ( all_ngrams )  # the ngram vocabulary size
    print v

    # a standard NLP dictionay (ngram -> index map) use to update the
    # one-hot vector p
    dic=dict ( (t, i) for i, t in enumerate ( all_ngrams ) )

    # for each class we create calculate a ratio (r_c)
    for c in counters.keys ( ):
        p_c=np.full ( v, alpha )
        counter=counters[ c ]

        for t in all_ngrams:
            p_c[ dic[ t ] ]+=counter[ t ]

        # normalize (l1 norm)
        p_c/=np.linalg.norm ( p_c, ord=1 )
        ratios[ c ]=np.log ( p_c / (1 - p_c) )
    return dic, ratios, v


def count_lines(data_file):
    """Counts the number of lines in a file
    """
    with open ( data_file ) as f:
        lines=f.readlines ( )
        return len ( lines )


def to_msvm(single_text, dic, v, ratios, grams, converter, classes):
    X, data, indptr, indices={}, {}, [ 0 ], [ ]
    grams=[ int ( i ) for i in grams ]
    for c in classes:
        data[ c ]=[ ]

    ngrams=converter.convert_text ( single_text, grams )
    for g in ngrams:
        if g in dic:
            index=dic[ g ]
            indices.append ( index )
            for c in classes:
                data[ c ].append ( ratios[ c ][ index ] )
    indptr.append ( len ( indices ) )

    for c in classes:
        X[ c ]=csr_matrix ( (data[ c ], indices, indptr), shape=(1, v), dtype=np.float32 )

    return X


def load_data(data_path, text_row, class_row, dic, v, ratios, grams, class2id, converter):
    """Create Train or Test matrix and Ground Truth Array
    """
    n_samples=count_lines ( data_path )
    n_r=len ( ratios )
    classes=ratios.keys ( )
    Y_real=np.zeros ( n_samples, dtype=np.int64 )

    # One X (sample) matrix and binary Y (truth) per class
    X, Y, data, indptr, indices={}, {}, {}, [ 0 ], [ ]

    for c in classes:
        Y[ c ]=np.zeros ( n_samples, dtype=np.int64 )
        data[ c ]=[ ]
    text_src=read_text_src ( data_path )
    n=0
    for line in text_src:
        if len(line)<text_row+1:continue
        try:
            label, text=line[ class_row ], line[ text_row ]
        except ValueError or IndexError:
            print 'line error'
            continue
        t=class2id[ label ]
        for c in classes:
            Y[ c ][ n ]=int ( c == t )
        Y_real[ n ]=t
        ngrams=converter.convert_text ( text, grams )
        for g in ngrams:
            if g in dic:
                index=dic[ g ]
                indices.append ( index )
                for c in classes:
                    # X[c][n][idx] = ratios[c][idx]
                    data[ c ].append ( ratios[ c ][ index ] )
        indptr.append ( len ( indices ) )
        n+=1

    for c in classes:
        X[ c ]=csr_matrix ( (data[ c ], indices, indptr), shape=(n_samples, v),
                            dtype=np.float32 )

    return X, Y, Y_real










class MultiClassifier ( object ):
    def __init__(self, name, grams="12",method=LogisticRegression, custom_tokenize=zh_tokenize):
        self.name=name
        self.method=method
        self.train_svm_file='%s_train.svm' % name
        self.models=None
        self.params=None
        self.converter=TextPreProcess ( custom_tokenize )
        self.grams = [int(i) for i in grams]

    def train(self, train_file, text_row=1, class_row=0):
        grams=self.grams

        print '------------build counters----------------'
        counters, class2id=build_counters ( train_file, grams, text_row, class_row, self.converter )
        print '------------compute_ratios----------------'
        dic, ratios, v=compute_ratios ( counters )
        classes=ratios.keys ( )
        print '-------------load data--------------------'
        Xs_train, Ys_train, y_train=load_data ( train_file, text_row, class_row, dic, v, ratios, grams, class2id,
                                                self.converter )
        print '-------------train classfier--------------------'
        svms=dict ( )
        for c in classes:
            svms[ c ]=self.method ( )
            # svms[c] = LogisticRegression()
            svms[ c ].fit ( Xs_train[ c ], Ys_train[ c ] )
        self.models=svms

        self.params=fill_config ( dic, ratios, v, class2id, text_row, class_row )

    def predict(self, single_text):
        # print '------------------predict-------------------'
        class2id=self.params[ 'class2id' ]
        classes=class2id.values ( )
        id2class={_id: _class for _class, _id in class2id.items ( )}

        Xs_test=to_msvm ( single_text, self.params[ 'dic' ], self.params[ 'v' ], self.params[ 'ratios' ],
                          self.grams, self.converter, classes )

        preds=dict ( )  # predic class
        probs=dict ( )
        for c in classes:
            preds[ c ]=self.models[ c ].decision_function ( Xs_test[ c ] )
            if hasattr ( self.models[ c ], "predict_proba" ):
                probs[ c ]=self.models[ c ].predict_proba ( Xs_test[ c ] )[ :, 1 ]
            else:
                probs[ c ]=(preds[ c ] - preds[ c ].min ( )) / (preds[ c ].max ( ) - preds[ c ].min ( ))

        max_score=float ( '-inf' )
        pred, prob=classes[ 0 ], 0.5
        for c in classes:
            if preds[ c ][ 0 ] > max_score:
                max_score=preds[ c ][ 0 ]
                pred=c
                prob=probs[ c ][ 0 ]
        return id2class[ pred ], prob







    def test(self, test_file, class_row=0, text_row=1):
        print '-------------------test---------------------------------'
        Xs_test, _, y_true=load_data ( test_file, text_row,
                                             class_row, self.params[ 'dic' ],
                                             self.params[ 'v' ], self.params[ 'ratios' ],
                                             self.grams, self.params[ 'class2id' ],
                                             self.converter )

        classes=self.params[ 'class2id' ].values ( )
        preds=dict ( )  # predic class
        probs=dict ( )  # predict probablity
        for c in classes:
            preds[ c ]=self.models[ c ].decision_function ( Xs_test[ c ] )
            if hasattr ( self.models[ c ], "predict_proba" ):
                probs[ c ]=self.models[ c ].predict_proba ( Xs_test[ c ] )[ :, 1 ]
            else:
                probs[ c ]=(preds[ c ] - preds[ c ].min ( )) / (preds[ c ].max ( ) - preds[ c ].min ( ))

        # not calculate the argmax
        pred=np.zeros ( len ( y_true ) )
        prob=np.zeros ( len ( y_true ) )
        for idx in range ( 0, len ( y_true ) ):
            max_score=float ( '-inf' )
            for c in classes:
                if preds[ c ][ idx ] > max_score:
                    max_score=preds[ c ][ idx ]
                    pred[ idx ]=c
                    prob[ idx ]=probs[ c ][ idx ]
        acc_svm=accuracy_score ( y_true, pred )
        print('NBSVM: %f' % (acc_svm,))
        return TestResult ( y_true, pred, prob, self.params[ 'class2id' ] )

    def save(self):
        if not os.path.exists ( self.name ):
            os.mkdir ( self.name )
        joblib.dump ( self.models, self.name + '/models.pkl' )
        cPickle.dump ( self.params[ 'ratios' ], open ( self.name + '/ratios.pickle', 'wb' ), -1 )
        cPickle.dump ( {'v': self.params[ 'v' ]}, open ( self.name + '/v.pickle', 'wb' ), -1 )
        cPickle.dump ( self.params[ 'class2id' ], open ( self.name + '/class2id.pickle', 'wb' ), -1 )
        cPickle.dump ( self.params[ 'dic' ], open ( self.name + '/dic.pickle', 'wb' ), -1 )
        self.converter.save ( self.name + '/id2tok.pickle' )

    def load(self, path=None):
        print '---------------load classifier-----------------'
        if path is not None:
            dest_src=os.path.join ( path, self.name )
        else:
            dest_src=self.name
        self.models=joblib.load ( dest_src + '/models.pkl' )
        ratios=cPickle.load ( open ( dest_src + '/ratios.pickle', 'rb' ) )
        v=cPickle.load ( open ( dest_src + '/v.pickle', 'rb' ) )[ 'v' ]
        dic=cPickle.load ( open ( dest_src + '/dic.pickle', 'rb' ) )
        class2id=cPickle.load ( open ( dest_src + '/class2id.pickle', 'rb' ) )
        self.params=fill_config ( dic, ratios, v, class2id )
        self.converter.load ( dest_src + '/id2tok.pickle' )
        return self



def dowork(para):
    model, inputs = para['model'], para['inputs']
    test_result = model.test(inputs)
    print test_result.accuracy_labels
    print test_result.recall_labels
    print test_result.accuracy_score
    test_result.show_result()


if __name__ == '__main__':
    number='9'
    d='./sample-data/train-%s-zh.txt' % number
    t='./sample-data/test-%s-zh.txt' % number
    # t = './sample-data/test-6-zh.txt'
    from pure_url_classify import url_tokenize
    mc=MultiClassifier ( 'test-%s-topic' % number, "12", custom_tokenize=url_tokenize )
    # d = './sample-data/train-jc.txt'
    # t = './sample-data/test-jc.txt'
    # mc = MultiClassifier('test-8-topic', custom_tokenize=zh_tokenize)
    import time

    start=time.time ( )
    # mc.train ( d )
    # mc.save ( )
    mc.load ( )

    # print mc.models[0].coef_
    # print mc.models[0].get_params()
    # print mc.models[0].densify()

    test_result=mc.test ( t )
    # cost=time.time ( ) - start
    # print 'cost time:', cost
    # print test_result.accuracy_labels
    # print test_result.recall_labels
    # print test_result.accuracy_score
    # test_result.show_result ( )
    # print mc.params[ 'class2id' ]
#     print mc.predict('藏头密码：73胜勇士全解读')
#     print mc.predict('金州勇士，73胜9负NBA历史上的常规赛最佳战绩。')
#     print mc.predict('解放军报评军队选手表现:男篮进前八大郅功不可没')
#     h="""北京时间2015年6月11日，根据ESPN的报道，有鉴于戴拉维多瓦在总决赛的优异表现，他本人球衣的销量也在疯狂攀升。
# 根据美国最大的正规体育用品零售商“Fanatics”的消息，戴拉维多瓦球衣在该公司的销量已经成为了NBA球员球衣里的销量冠军（best-seller）。
# 并且截至目前，戴拉维多瓦总决赛期间的球衣销量已经超过了其整个常规赛的售出总额。
# 从当地时间6月9日下午4点到6月10日下午4点，美国有来自20个不同州的用户在网上订购了戴拉维多瓦的球衣
#     """
#     print mc.predict(h)
    # print mc.predict('14英尺外 转身 后仰 打板 跳投 不中')
    # print mc.predict('助攻 反身 扣篮 得2分')
    # print mc.predict('26 英尺外 三分 跳投 不中')
    # print mc.predict('中国“铁腰”与英超球队埃弗顿分道扬镳，闪电般转投谢联')


