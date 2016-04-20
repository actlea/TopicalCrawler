#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: tmsvm_util.py
@time: 16-3-17 上午9:43
@description:使用了tmsvm, tgrocery
"""

from base import *



config = {
    'svm_type':'libsvm',
    'global_fun':'idf',
    'local_fun':'tf',
    'ratio':0.5,
    'learner_opts':''
}

#=========================================
#特征选择
import math
import measure
import cPickle




class Chi(object):
    def __init__(self):
        self.global_weight_dic = {}


    def dic_add(self, dic, key):
        if key in dic:
            dic[key] += 1.0
        else:
            dic[key] = 1.0

    def build_chi_dic(self, tables, class2id, tok_dic):
        """
        :param tables: [(label, tokids)]
        :return:
        chi_dic: chi_dic[tid][label]=n
        cat_num: number of each class
        """
        cat_num_dic = {}
        #compute idf
        # chi_dic = [[0.0 for i in class2id] for j in tok_dic]
        chi_dic = {j:{i:0.0 for i in class2id} for j in tok_dic.values()}
        for line in tables:
            label, tids = line
            self.dic_add(cat_num_dic,label)
            for tid in list(set(tids)):
                chi_dic[tid][label] += 1.0
        return chi_dic, cat_num_dic

    def chi_max_score(self, chi_dic, cat_num_dic):

        """ 利用卡方公式计算每个term的分值
        w(t,c)=(AD-BC)^2/(A+B)(C+D)
        :param chi_dic:  chi_dic[tid][label]=idf
        :param cat_num_dic: {class:number}
        :return: chi_score_dic
        """
        chi_score_dic={}

        for tid in chi_dic:
            sum_tid = float(sum(chi_dic[tid].values()))
            sum_class = float(sum(cat_num_dic.values()))
            chi_score = 0.0
            for label in cat_num_dic:
                A = chi_dic[tid][label]
                B = cat_num_dic[label]-A
                C = sum_tid - A
                D = sum_class - sum_tid - cat_num_dic[label] + A
                if (A+B)*(C+D) == 0:
                    chi_score=0
                else:
                    chi_score = max(chi_score,math.pow(A*D-B*C,2)/((A+B)*(C+D)))
            chi_score_dic[tid] = chi_score
        return chi_score_dic

    def chi_feature_select(self, tables, global_fun, ratio, class_dic, tok_dic):
        """最终会选择top $ratio 的作为最终的词典
        """
        chi_dic, cat_num_dic = self.build_chi_dic(tables, class_dic, tok_dic)

        chi_score_dic = self.chi_max_score(chi_dic, cat_num_dic)
        sorted_keys=sorted(chi_score_dic.items(),key=lambda x:x[1],reverse=True)

        temp_dic = measure.global_f(global_fun)(chi_dic, cat_num_dic,len(tables))

        res = sorted_keys[:int(len(chi_score_dic)*ratio)]

        self.global_weight_dic = {tid:temp_dic[tid] for tid,_ in res}
        # from sklearn.feature_selection import SelectKBest
        # from sklearn.feature_selection import chi2
        # X = [label[0] for label in tables]
        # y = [label[1] for label in tables]
        # data  = SelectKBest(chi2, k=ratio*len(tok_dic)).fit_transform(X, y)
        # self.global_weight_dic = data

    def save(self, dest_dir):
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        weight_path = os.path.join(dest_dir,'g_weight_dic.pickle')
        cPickle.dump(self.global_weight_dic, open(weight_path, 'wb'), -1)


    def load(self, src_dir):
        src_file = os.path.join(src_dir,'g_weight_dic.pickle')
        tok_dic_path = os.path.join(src_dir,'tok2id.pickle')
        self.global_weight_dic = cPickle.load(open(src_file, 'rb'))
        self.tok2id = cPickle.load(open(tok_dic_path, 'rb'))
        return self




class TMSVM(object):
    def __init__(self, name, config=None, custom_tokenize=None):
        if config is not None:
            self.config = config
            self.config['name'] = name
        self.name = name

        self.chi = None
        self.learner_model = None
        self.classifier = None
        self.train_svm_file = '%s_train.svm' % self.name
        if custom_tokenize is None:
            self.custom_tokenize = term_seg
        else:
            self.custom_tokenize = custom_tokenize

        self.tok2id = None
        self.class2id = None

    def train(self, train_src, delimiter='\t'):
        print '--------------------分词-------------------'
        text_src = read_text_src(train_src, delimiter)
        print '----------------------生成词典-----------------'
        tok_dic, class2id, tables = gen_term_cat__dic(text_src, self.custom_tokenize)
        print '----------------------特征选择-----------------'
        self.chi = Chi()
        self.chi.chi_feature_select(tables, config['global_fun'], config['ratio'], class2id, tok_dic)
        print '----------------------创建训练的参数-----------------'
        convert(tables, self.chi.global_weight_dic, class2id, config['local_fun'], self.train_svm_file)
        print '----------------------训练--------------------------'
        self.learner_model = train(config, self.train_svm_file)
        self.tok2id, self.class2id = tok_dic, class2id

    def predict(self, single_text):
        if self.learner_model is None:
            raise Exception('This model is not usable because svm model is not given')
            # process unicode type
        if isinstance(single_text, unicode):
            single_text = single_text.encode('utf-8')
        if not isinstance(single_text, str):
            raise TypeError('The argument should be plain text')

        feat = to_svm(preprocess(single_text, self.custom_tokenize, self.tok2id), self.chi.global_weight_dic,
                      self.config['local_fun'])
        y, dec = predict_one(feat, self.learner_model)
        id2class = {_id:_class for _class,_id in self.class2id.items()}
        y = id2class[int(y)]
        return y, dec

    def test(self, text_src, delimiter='\t'):
        text_src = read_text_src(text_src, delimiter)
        true_y = []
        predicted_y = []
        probs = []
        for line in text_src:
            try:
                label, text = line
            except ValueError:
                continue
            y, dec = self.predict(text)
            probs.append(dec)
            predicted_y.append(y)
            true_y.append(label)
        return TestResult(true_y, predicted_y, probs)


    def save(self, force=False):
        if not os.path.exists(self.name):
            os.mkdir(self.name)

        self.learner_model.save(self.name+'/learner', force)
        self.chi.save(self.name+'/dic')

        tok_dic_path = os.path.join(self.name+'/dic','tok2id.pickle')
        cPickle.dump(self.tok2id, open(tok_dic_path, 'wb'), -1)
        class2id_path = os.path.join(self.name+'/dic','class2id.pickle')
        cPickle.dump(self.class2id, open(class2id_path, 'wb'), -1)

        pickle_write(self.tok2id, self.chi.global_weight_dic, self.name+'/dic/tok_id_weight.txt')


    def load(self):
        learner = LearnerModel(self.config)
        self.learner_model = learner.load(self.name+'/learner')
        chi = Chi()
        self.chi = chi.load(self.name+'/dic')

        tok_dic_path = os.path.join(self.name+'/dic','tok2id.pickle')
        self.tok2id = cPickle.load(open(tok_dic_path, 'rb'))
        class2id_path = os.path.join(self.name+'/dic','class2id.pickle')
        self.class2id = cPickle.load(open(class2id_path, 'rb'))


def tokenize(text):
    return text.split()


if __name__ =='__main__':
    c = TMSVM('test-zhb-tmsvm', config)
    d = './sample-data/train.txt'
    t = './sample-data/test.txt'
    c.train(d)
    c.save(force=True)
    c.load()

    test_result = c.test(t)
    print test_result.accuracy_labels
    print test_result.recall_labels
    print test_result.accuracy_score
    test_result.show_result()














