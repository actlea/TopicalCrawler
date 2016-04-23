#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: text.py
@time: 16-3-3 上午9:11
@reference tgrocery and tmsvm
"""
import re
import string
import os

from util import FileHelper

# resource_dir = 'D:\\OtherWork\\PythonProject\\TopicalCrawl\\TopicalCrawl\\'
resource_dir = '/mnt/UbutunShare/TopicalCrawl/TopicalCrawl'

class StopWords(object):

    PUNCTUATION = re.compile(
        "[^\\p{Ll}\\p{Lu}\\p{Lt}\\p{Lo}\\p{Nd}\\p{Pc}\\s]")
    # TRANS_TABLE = str.maketrans('', '')
    _cached_stop_words = {}

    def __init__(self, language='en'):
        path = os.path.join('text', 'stopwords-%s.txt' % language)
        if language not in self._cached_stop_words:
            self._cached_stop_words[language] = \
                set(FileHelper.loadResourceFile(path).splitlines())
        self.STOP_WORDS = self._cached_stop_words[language]
        self.language = language

    def remove_punctuation(self, content):
        # code taken form
        # http://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        content_is_unicode = isinstance(content, unicode)
        if content_is_unicode:
            content = content.encode('utf-8')

        zh_punc = u',?、。“”《》！，：；？＞＜＠＃＄％︿＆＊＋～｜［］｛｝-'
        zh_punc_list = [i for i in zh_punc]
        en_punc_list = [unicode(c) for c in string.punctuation]

        zh_punc_list.extend(en_punc_list)

        trans_table = {ord(c): None for c in zh_punc_list}
        stripped_input = content.decode('utf-8').translate(trans_table)

        return stripped_input

    def candidate_words(self, stripped_input):
        return stripped_input.split(' ')


import jieba
# import jieba.analyse as analyse


def load_stopwords(stopword_path):
    import jieba.analyse as analyse
    stopwords_path = os.path.join(resource_dir, 'resources', 'text', 'stopwords-zh.txt')
    analyse.set_stop_words(stopwords_path)


def load_keywords(dic_path):
    jieba.load_userdict(dic_path)


# import jieba.analyse as analyse



# jieba.load_userdict(user_dict_path)
# analyse.set_stop_words(stopwords_path)

class StopWordsChinese(StopWords):
    """Chinese segmentation
    """
    def __init__(self, language='zh'):
        super(StopWordsChinese, self).__init__(language='zh')

    def candidate_words(self, stripped_input):
        # jieba builds a tree that takes a while. avoid building
        # this tree if we don't use the chinese language

        return jieba.cut(stripped_input, cut_all=True)


def seg_text(content):
    return jieba.cut(content, cut_all=True)





class ParseContent(object):

    @staticmethod
    def tf_if_parse(content, keywords=None, topK = 50):
        """ keywords must be include
        """
        import jieba.analyse as analyse
        import jieba

        tfidf_path = os.path.join(resource_dir,'resources','text','tf-idf.txt')

        user_dict_path = os.path.join(resource_dir,'resources','text','user_dict.txt')
        stopwords_path = os.path.join(resource_dir,'resources','text','stopwords-zh.txt')

        jieba.load_userdict(user_dict_path)
        analyse.set_stop_words(stopwords_path)
        analyse.set_idf_path(tfidf_path)
        tags = analyse.extract_tags(content, topK=topK)
        return tags


if __name__ == '__main__':
    with open('1.html', 'r') as f:
        content = f.read()
    P = ParseContent()
    res = P.tf_if_parse(content)
    print '/'.join(res)
