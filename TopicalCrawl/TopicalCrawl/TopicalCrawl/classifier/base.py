#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: base.py
@time: 16-3-17 上午9:46
@description:
"""
from TopicalCrawl.text import seg_text
from TopicalCrawl.dependence import porter2
import math
import shutil
import os
import cPickle
import measure


# stopword_path = 'D:\\OtherWork\\PythonProject\\TopicalCrawl\\TopicalCrawl\\resources\\text\\stopwords-zh.txt'
stopword_path = '/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/resources/text/stopwords-zh.txt'

def read_stopword_dic(stopword_file):
    """ term:index
    """
    f= file(stopword_file,'r')
    dic={}
    for line in f.readlines():
        line = line.split('\t')
        if len(line)<1: continue
        if line[0].strip() not in dic:
            dic[line[0].strip()] = len(dic)
    f.close()
    return dic

def trans(word, stopword_dict):
    if isinstance(word, unicode):
        word = word.decode('utf-8')

    trans_table = {ord(c): None for c in stopword_dict}
    stripped_input = word.translate(trans_table)
    return stripped_input

def term_seg(text, stopwords_file=stopword_path):
    """分词，去除停等词之后，使用str_splitTag连接词
    """
    seg_text_list = seg_text(text)

    #去除停等词
    stopword_dic = read_stopword_dic(stopwords_file)
    seg_text_list = [term for term in seg_text_list if term.encode('utf-8', 'ignore') not in stopword_dic]
    result = seg_text_list

    return result

def stem_words(word):
    return porter2.stem(word).lower()

def read_text_src(text_src, delimiter='\t'):
    if isinstance(text_src, str):
        with open(text_src, 'r') as f:
            text_src = [line.split(delimiter) for line in f]
    elif not isinstance(text_src, list):
        raise TypeError('text_src should be list or str')
    return text_src


def is_bad_tok(tok):
    """
    :param tok: tok is unicode
    :return:
    """
    if not isinstance(tok, unicode):
        tok = tok.decode('utf-8')
    tok = tok.strip()
    if len(tok)==0: return False

    for i in tok:
        if (u'z'>=i>=u'a' or u'Z'>=i>=u'A') or (u'\u2E80' <=i<= u'\u9fff'):
            return True
    return False






def gen_term_cat__dic(text_src, custom_tokenize):
    "生成类别和词项词典"
    tok2id = {}
    class2id = {}
    tables = []
    for line in text_src:
        try:
            label, text = line
        except ValueError:
            continue
        if label not in class2id:
            class2id[label] = len(class2id)
        toks = custom_tokenize(text)
        toks = [stem_words(tok.strip()) for tok in toks if is_bad_tok(tok)]

        ret = []
        for tok in toks:
            if tok not in tok2id:
                tok2id[tok] = len(tok2id)
            ret.append(tok2id[tok])
        tables.append((label, ret))
        # for x, y in zip(toks[:-1], toks[1:]):#ToDo:2016/03/23
        #     if (x, y) not in tok2id:
        #         tok2id[x, y] = len(tok2id)
        #     ret.append(tok2id[x,y])
        # tables.append((label, ret))

    return tok2id, class2id, tables


def preprocess(text_src, custom_tokenize, tok_dic=None):
    toks = custom_tokenize(text_src)
    toks = [stem_words(tok.strip()) for tok in toks if is_bad_tok(tok)]
    ret = []
    for tok in toks: #ToDo:2013/03/23
        if tok in tok_dic:
            ret.append(tok_dic[tok])
    # for x, y in zip(toks[-1:], toks[1:]):
    #     if (x, y) in tok_dic:
    #         ret.append(tok_dic[x,y])
    return ret



def to_svm(tids, global_weight_dic, local_fun, class_id=None):
    """ 根据词典和权重词典构造svm分类所需的输入格式
    :param tids:
    :param tok_dic: 词典
    :param class2id: 类映射
    :param global_weight_dic:权重词典
    :return: feat
    """
    local_fun = measure.local_f(local_fun)
    feat = {}

    #buidl feature vector
    for tid in tids:
        if tid in global_weight_dic:
            if tid in feat:
                feat[tid] += 1.0
            else:
                feat[tid] = 1.0

    #compute feature weight
    for tid,weight in feat.items():
        feat[tid] = 1.0 * local_fun(weight) * global_weight_dic[tid]
    #normalize
    vec_sum = sum([weight**2.0 for weight in feat.values()])

    vec_length = math.sqrt(vec_sum)
    if vec_length!=0:
        for tok, weight in feat.items():
            feat[tok] = 1.0*weight/vec_length
    if class_id is not None:
        return feat,class_id
    else:
        return feat


def convert(tables, global_weight_dic, class2id, local_fun, output):
    with open(output, 'w') as w:
        for line in tables:
            label, tids = line
            feat, class_id = to_svm(tids, global_weight_dic, local_fun, class2id[label])
            w.write('%s %s\n' % (str(class_id), ''.join(' {0}:{1}'.format(f, feat[f]) for f in sorted(feat))))




import sys
class TrainMethod(object):
    def __init__(self, svmtype):
        self.svm_type = svmtype
        self.set_svm_type()

    def set_svm_type(self):
        from TopicalCrawl.dependence import svmutil, liblinearutil
        if self.svm_type == 'libsvm':
            self.read_problem = svmutil.svm_read_problem
            self.train = svmutil.svm_train
            self.predict = svmutil.svm_predict
            self.save_model = svmutil.svm_save_model
            self.load_model = svmutil.svm_load_model
            self.get_labels = svmutil.get_labels
            self.get_nr_class = svmutil.get_nr_class
        elif self.svm_type=="liblinear":
            self.read_problem = liblinearutil.svm_read_problem
            self.train = liblinearutil.train
            self.predict = liblinearutil.predict
            self.save_model = liblinearutil.save_model
            self.load_model = liblinearutil.load_model
            self.get_labels = liblinearutil.get_labels
            self.get_nr_class = liblinearutil.get_nr_class
        else:
            sys.exit('svm type set error')

    def classer_value(self, values):
        '''计算类得隶属度,libsvm采用的为one-against-one算法。
        liblinear采用的为oen-against-rest算法。因此在计算最终的隶属度分数上有所区别.
        计算公式为：sum(vi)/(2*k)+k/(2*n):n为所有参数类得总数,对libsvm为all-1,liblinear为1，k为支持该类的数,
        vi为支持该类的value
        '''
        svm_type = self.svm_type
        if svm_type=="libsvm":
            n = 1+int(math.sqrt(2*len(values)+1))
            size = n-1
            vote=[0]*n
            score=[0]*n
            p=0
            for i in range(n-1):
                for j in range(i+1,n):
                    if values[p]>0:
                        vote[i]+=1
                        score[i]+=math.fabs(values[p])
                    else :
                        vote[j]+=1
                        score[j]+=math.fabs(values[p])
                    p+=1
            mindex = 0
            for i in range(1,n):
                if vote[i]>vote[mindex]:
                    mindex = i
            k = vote[mindex]
            init_score=score[mindex]
            return float(init_score)/(2.0*k)+float(k)/(2.0*size)
        elif svm_type=="liblinear":
            n =len(values)
            mindex = 0
            for i in range(1,n-1):
                if values[i]>values[mindex]:
                    mindex = i
            size = 1
            k = 1
            init_score = values[max]
            return float(init_score)/(2.0*k)+float(k)/(2.0*size)
        else:
            return 0.0



class LearnerModel(object):
    def __init__(self, config=None, svm_model=None):
        self.method = TrainMethod(config['svm_type'])
        self.svm_model = svm_model
        self.config = config

    def save(self, dest_dir, force=False):

        if os.path.exists(dest_dir):
            if force:
                shutil.rmtree(dest_dir)
            else :
                raise OSError('Please use force option to overwrite the existing files.')
        else:
            os.mkdir(dest_dir)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        model_path = os.path.join(dest_dir,'model')
        self.method.save_model(model_path, self.svm_model)
        config_path = os.path.join(dest_dir,'config.pickle')
        cPickle.dump(self.config, open(config_path,'wb'), -1)

    def load(self, src_dir):
        model_path = os.path.join(src_dir,'model')
        config_path = os.path.join(src_dir,'config.pickle')
        self.svm_model = self.method.load_model(model_path)
        config = cPickle.load(open(config_path, 'rb'))
        self.method = TrainMethod(config['svm_type'])
        return self

class TestResult(object):
    def __init__(self, true_y, predicted_y, prob, class2id):
        from sklearn.metrics import accuracy_score
        self.id2class = {_id:_class for _class,_id in class2id.items()}
        self.true_y = true_y
        self.predicted_y = predicted_y
        self.prob = prob
        self.accuracy_score = accuracy_score(self.true_y, self.predicted_y)
        self._compute_accuracy_overall()
        self._compute_accuracy_recall_labels()

    def _compute_accuracy_overall(self):
        l = len(self.true_y)
        self.accuracy_overall = sum([self.true_y[i] == self.predicted_y[i] for i in range(l)]) / float(l)

    def _compute_accuracy_recall_labels(self):
        labels = {}
        for idx, predicted_label in enumerate(self.predicted_y):
            true_label = self.true_y[idx]
            if predicted_label not in labels:
                labels[predicted_label] = [0, 0, 0]
            if true_label not in labels:
                labels[true_label] = [0, 0, 0]
            if predicted_label == true_label:
                labels[predicted_label][0] += 1
            labels[predicted_label][1] += 1
            labels[true_label][2] += 1
        self.accuracy_labels = {}
        self.recall_labels = {}
        for key, val in labels.iteritems():
            try:
                self.accuracy_labels[key] = float(val[0]) / val[1]
            except ZeroDivisionError:
                self.accuracy_labels[key] = float(0)
            try:
                self.recall_labels[key] = float(val[0]) / val[2]
            except ZeroDivisionError:
                self.recall_labels[key] = float(0)

    @staticmethod
    def draw_table(data, row_labels, column_labels, id2class):
        row_format = '{:<15}' * (len(column_labels) + 1)
        table_string = '%s\n' % row_format.format('', *column_labels)
        for row_label, row_data in zip(row_labels, data):
            table_string += '%s\n' % row_format.format(id2class[row_label], *row_data)
        return table_string

    def show_result(self):
        print self.draw_table(
            zip(
                ['%.2f%%' % (s * 100) for s in self.accuracy_labels.values()],
                ['%.2f%%' % (s * 100) for s in self.recall_labels.values()]
            ),
            self.accuracy_labels.keys(),
            ('accuracy', 'recall'), self.id2class
        )

    def __str__(self):
        return str(self.accuracy_overall)


def predict_one(xi, m):
    p_label, p_score = m.method.predict([xi], m.svm_model)
    return p_label[0], m.method.classer_value(p_score[0])



def train(config, train_data_file):
    method = TrainMethod(config['svm_type'])
    prob, params = method.read_problem(train_data_file)
    model = method.train(prob, params, config['learner_opts'])
    return LearnerModel(config, model)


def pickle_write(tok2id, global_weight_dic, outfile):
    with open(outfile, 'w') as fw:
        id2tok = {tid:tok for tok, tid in tok2id.items()}
        for i in range(len(tok2id)):
            weight = global_weight_dic.get(i,1.0)
            try:
                tok = id2tok[i]
                if isinstance((id2tok[i]), unicode):
                    tok = id2tok[i].encode('utf-8','ignore')
                fw.write('%s\t%d\t%.3f\n' %(tok, i, weight))
            except UnicodeDecodeError:
                continue










