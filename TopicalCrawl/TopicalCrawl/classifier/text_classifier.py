#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: text_classifier.py
@time: 16-4-22 上午11:22
"""

from util import EncodeHelper
from base import *
from collections import Counter

from TopicalCrawl.htmlParse import *

football_path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/resources/pos_neg_dic/football.txt'



def is_in_dic(words, dic):
    for w in words:
        if w in dic:
            return True
    return False


def anchor_predict(anchor_text,
                   model,
                   pos_label='0',
                   neg_label='1',
                   pos_words=None):
    '''
    预测锚文本是否与主题先关
    :param anchor_text: 锚文本
    :param model: 短文本分类模型
    :param pos_label: 正样本标签
    :param neg_label: 负样本标签
    :param pos_dic: 正样本词典
    :return: 预测标签，预测概率
    '''
    if anchor_text is None or len(anchor_text)<2:
        return pos_label, 0.0
    toks = zh_tokenize(anchor_text)
    if pos_words is not None:
        if is_in_dic(toks, pos_words):
            return pos_label,1.0
    pred, prob = model.predict(anchor_text)
    return pred, prob


def para_predict(para_text,
                 model,
                 pos_label='0',
                 neg_label='1'):
    '''预测正文文本段'''
    if para_text is None :
        return pos_label, 0.0

    if not isinstance(para_text, unicode):
        para_text = para_text.decode('utf-8')

    if len(para_text)<10:
        return pos_label, 0.0
    pred, prob = model.predict(para_text)
    return pred, prob


def page_predict(title,
                      headlines,
                      meta,
                      para,
                      url_result,
                      short_text_model,
                      long_text_model,
                      pos_words=None,
                      pos_label='0',
                      neg_label='1'):
    '''
    预测网页所属的类型
    :param title: 大标题
    :param headlines: 小标题
    :param meta: meta内容
    :param para: 正文内容
    :param url_result:url的预测结果
    :param short_text_model: 短文本分类器
    :param long_text_model: 长文本分类器
    :param pos_label:
    :param neg_label:
    :return: 长文本分类器
    '''
    pred_t, prob_t = anchor_predict(title, short_text_model, pos_label, neg_label, pos_words)

    pred_h, prob_h = anchor_predict(headlines, short_text_model, pos_label, neg_label, pos_words)

    pred_m, prob_m = anchor_predict(meta, short_text_model, pos_label, neg_label, pos_words)

    pred_p, prob_p = para_predict(para, long_text_model, pos_label, neg_label)

    atom=[(pred_t, prob_t),(pred_h, prob_h),(pred_m, prob_m),(pred_p, prob_p),url_result]

    # 过滤掉概率值小于0.5的预测结果
    result=[p for p in atom if p[1]>=0.5]

    labels=[p[1] for p in result]

    label_probs={}
    for r in result:
        label_probs.setdefault(r[0],[] ).append(r[1])


    counter = Counter(labels).most_common()


    res_label = counter[0][0]
    res_prob = max(label_probs[res_label])

    return res_label, res_prob




def parse_page(response, encoding):
    '''
    parse page and get document struct
    :param response:
    :param encoding:
    :return:
    '''
    url = response.url
    body = response.body
    encoding = response.encoding
    try:
        content = EncodeHelper.get_unicode(body, encoding)
        if content is None:
                return
        content = content.encode('utf-8')
    except UnicodeError:
        return None

    # document = html2words(content, url)
    # return document





