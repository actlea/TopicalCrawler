#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: actlea
@file: pure_url_classify.py
@time: 16-3-21 下午2:49
@description:
"""


from util import EncodeHelper
from base import *
from urllib import unquote
import re

None_Split_Tokens=[ 'http', 'https', 'www', 'com', 'cn', 'asp', 'ftp' ]

re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)", re.U)

SPORTS=['sport', 'nba', 'cba', 'f1']


def read_dic(path):
    with open ( path ) as fr:
        lines=fr.read ( )
    lines=lines.splitlines ( )
    words=[ word.strip ( ) for word in lines if len ( word.strip ( ) ) != 0 ]
    return list ( set ( words ) )


def url_decode(url):
    return unquote(url)


def stripNonAlphaNum(text):
    text = EncodeHelper.get_unicode(text)
    text=re.sub ( r'[^a-zA-Z0-9]', ' ', text )
    return text.encode ( 'utf-8' )

def stripNoneZh(text):
    if not isinstance(text, unicode):
        text = EncodeHelper.get_unicode(text)
    text = re.sub(re_han_default, '', text)
    return text


def right_max(i,dic, s):
    right = i
    for j in range(i, len(s)):
        if stem_words(s[i-1:j]) in dic:
            right = j
    return right


def get_left(i, dic, s):
    '''
    #在[0:i]之间寻找第一个有意义的词，作为left返回, 没有找到有意义的词，返回-1
    :param i:
    :param dic: 词典
    :param s: 单词
    :return:
    '''
    left = -1
    for j in range(i):
        if stem_words(s[j:i]) in dic:
            left = j
            break

    return left


def word_break(s, dic):
    if stem_words(s) in dic:
        return s

    for i in range(1, len(s)):
        #找到i左边第一个有意义的单词,
        left = get_left(i, dic, s)
        if left==-1:
            continue
        prefix = s[left:i]
        if prefix in dic:
            #找到i右边第一个有意义的单词
            j = right_max(left+1, dic, s)
            prefix = s[left:j]
            suffix=s[j:]
            segsuffix=word_break(suffix, dic)
            if segsuffix is not None:
                return prefix + ' ' + segsuffix
    return None



def url_tokenize(url, grams = '12',splitLen=4):
    """ 假如gram=4,http://www.allwatchers.com/Topics/Info_3922.asp,
    那么长度不超过4的如htpp, www,com等都直接保留，而长度超过4的allwatchers则会
    被分割为allw llwa, lwat, watc, ...等以4为长度的单位
    :param url:
    :param gram:
    :return:
    """
    dict_path='../resources/wordsEn.txt'
    stopword_path='../resources/stopwordEn.txt'
    pinYin_path = '../resources/pinYin.txt'
    word_dict=read_dic ( dict_path )
    stopwords=read_dic ( stopword_path )
    pinYin_dic = read_dic ( pinYin_path )

    #1.先处理英文
    url = url_decode(url)
    tmp=stripNonAlphaNum ( url )
    if tmp is None:
        return None
    toks= tmp.split ( ' ' )
    toks = [tok.strip() for tok in toks if len(tok.strip())!=0]
    ret = []
    for tok in toks:
        if len ( tok ) > splitLen and tok not in None_Split_Tokens:
            en_words = word_break(tok, word_dict)
            if en_words is not None:
                ret += en_words.split(' ')
            else:
                ret.append(tok)

            # whether it is pinyin
            # pinyin_words = word_break(tok, pinYin_dic) #ToDo 2016-04-19
            # if pinyin_words is not None:
            #     ret += pinyin_words.split(' ')
        else:
            ret.append(tok)
    #2.然后处理中文
    # zh_url = stripNoneZh(url)
    # toks = zh_tokenize(zh_url)
    ret += toks
    return ret


def url_tokenize2(url, grams = 3):
    """ 假如gram=4,http://www.allwatchers.com/Topics/Info_3922.asp,
    那么长度不超过4的如htpp, www,com等都直接保留，而长度超过4的allwatchers则会
    被分割为allw llwa, lwat, watc, ...等以4为长度的单位
    :param url:
    :param gram:
    :return:
    """
    url = url_decode(url)
    tmp=stripNonAlphaNum ( url )
    if tmp is None:
        return None
    toks= tmp.split ( ' ' )
    toks = [tok.strip() for tok in toks if len(tok.strip())!=0]
    ret = []
    for tok in toks:
        if len(tok) > grams and tok not in None_Split_Tokens:
            for i in range(len(tok)-int(grams)+1):
                ret += tok[i:i+grams]
    ret += toks
    return ret




re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)", re.U)

# # mc = MultiClassifier('test-1-url', custom_tokenize=url_tokenize())
# tok = TextPreProcess(url_tokenize)
# url = 'http://www.iallwatchers.com/Topics/Info_3922.asp'
# print tok.convert_text(url, '12')
# url = 'http://www.sogou.com/web?query=一岁闪亮登场&p=31210100&fhintidx=9'
# print tok.convert_text(url, '12')
# # print url_tokenize(url)
# url = 'http://roll.sports.com.cn/s_nanzuguojiadui_all/index.shtml'
# print tok.convert_text(url, '12')

# print stripNonAlphaNum(url_decode('http://www.baidu.com/s?wd=%ce%f7%bc%d7%c8%fc%b3%cc%b1%ed'))