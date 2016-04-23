#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: htmlParse.py
@time: 16-3-4 上午9:15
"""
import os
import copy
import sys
import cPickle
import json
from lxml.html import builder as E
from lxml.html import tostring as htmltostring
from items import Document, UrlItem

from api import HtmlHelper
from util import FileHelper, ElementHelper, StringHelper, EncodeHelper
from cetd import get_page_link_ratio, get_aricle_cetd
from url import collect_urls, get_link_word_by_pair, get_link_word_pairs_by_wish, url_hashcode
from TopicalCrawl.classifier.multiclassifier import MultiClassifier

import re

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


def build_html_tree(title_text, headlines, meta_description, main_para_text, base_url):
    """ 创建树结构
    :param title_text: 文章的标题
    :param headlines: 文章中的显著标题
    :param meta_description:
    :param main_para_text:
    :param base_url:
    :return:
    """

    def is_valid(c):
        return c is not None and len(c) > 0

    div=E.DIV()

    if is_valid(main_para_text):
        div.text=main_para_text

    # append headlines to div
    if is_valid(headlines):
        for line in headlines:
            div.append(E.H2(line))

    # set title
    title=E.TITLE(title_text)

    # set meta node
    meta=E.META()
    if is_valid(meta_description):
        meta.set('content', meta_description)

    # set base url
    base=E.BASE()
    if is_valid(base_url):
        base.set('href', base_url)

    # set body node
    body=E.BODY(div)

    htmltree=E.HTML(
        E.HEAD(
            title,
            meta,
            base
        ),
        body
    )

    html2string=htmltostring(htmltree, encoding='utf-8')

    return html2string


def encode(content):
    try:
        if isinstance(content, unicode):
            content=content.encode('utf-8')
        return content
    except UnicodeDecodeError:
        return ''


def html2words(docstring, base_url, encoding=None, supervisior=None):
    """
    从网页源码中抽取正文
    :param docstring:
    :param encoding:
    :return:
    """
    string_size=sys.getsizeof(docstring)
    byte_size=string_size / (1024)
    if byte_size < 1:
        return

    docstring=docstring.lower()
    doctree=HtmlHelper.create_doc(docstring, encoding)

    if doctree is None: return None
    copy_doc=copy.deepcopy(doctree)
    # try:
    #
    #     link_ratio=get_page_link_ratio(copy_doc)
    #     print 'link_ratio: %f' % link_ratio
    #
    #     if link_ratio > 0.6:
    #         print 'this is home page'
    #         return None
    # except ValueError:
    #     return None

    doctree=HtmlHelper.pre_process_domtree(doctree)
    if doctree is None:
        return None
    # get page title and para content
    para, title=HtmlHelper.get_article(doctree, debug=False)

    # get page meta keywords and meta description
    meta_description=HtmlHelper.get_meta_description(copy_doc)

    # get headlines in page
    cleaned_body=ElementHelper.get_body(doctree)
    headlines=HtmlHelper.get_headline_content_in_cleaned_body(cleaned_body)

    # get all urls
    url_items=[]
    for item in get_link_word_by_pair(docstring, base_url, supervisior): url_items.append(item)

    document=Document()
    document['base_url']=base_url
    document['title']=title
    document['meta']=meta_description
    document['headlines']=headlines
    document['para']=para
    document['url_items']=url_items


    return document


def save(document, json_file=None, dest_file=None):
    if dest_file is not None:
        html=build_html_tree(document['title'], document['headlines'],
                             document['meta_descrip'], document['para'],
                             document['base_url'])
        FileHelper.WriteInUTF8(dest_file, html)
    if json_file is not None:
        json.dump({'title': document['title'], 'meta': document['meta_descrip'], 'url': document['base_url'],
                   'div': document['para']}, open(json_file, 'w'))



def save_url(url_item, dest_file):
    cPickle.dump(url_item, open(dest_file, 'wb'), -1)


def gen_url_item(url, title, label, precision):
    item=UrlItem()
    item['url']=url
    item['anchor_text']=title
    item['label']=label
    item['interestness']=precision
    return item


def get_history(history):
    if os.path.exists(history):
        with open(history, 'r') as f:
            visited_files=f.read()
            visited_files = visited_files.splitlines()
    else:
        visited_files=[]
    return visited_files


def init(name, model='test-zh-topic'):
    src_dir='/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/%s' % name

    supervisior=MultiClassifier('test-2-topic')
    supervisior.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')

    history=os.path.join(src_dir, 'history.txt')  # 存储已经处理过的文件
    tree=os.path.join(src_dir, 'tree')  # 存储经过解析之后的树结构
    json_dir=os.path.join(src_dir, 'json')
    original=os.path.join(src_dir, 'original')  # 网页源文件
    url=os.path.join(src_dir, 'url')  # 存储从网页中抽取的url item,没有经过预测



    url_sample_file=os.path.join(src_dir, 'url.sample.txt')  # 存储经过预测和标记之后的url item
    url_sample_pickle=os.path.join(src_dir, 'url.sample.pickle')
    url_sample={}
    document_class={}
    dcoument_class_file=os.path.join(src_dir, 'document.class.json')  # 存储经过分类之后的网页

    if not os.path.exists(tree):
        os.mkdir(tree)
    if not os.path.exists(url):
        os.mkdir(url)
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)

    return supervisior, original, history, tree, json_dir, url, url_sample_file, \
           url_sample_pickle, url_sample,document_class, dcoument_class_file



def create_train_samples(name, model='test-zh-topic'):

    supervisior, original, history, tree, json_dir,\
    url, url_sample_file, url_sample_pickle, url_sample,\
    document_class, dcoument_class_file = init(name)

    visited_files = get_history(history)
    finput=open(history, 'a')

    list_file = os.listdir(original)
    num = lambda x: int(x.split('.')[0])
    list_file.sort(cmp=lambda a,b:num(a)-num(b))

    for f in list_file:
        if f in visited_files:
            continue
        print '========================================'
        print 'parse file: %s ....' % f
        t=os.path.join(original, f)

        content=FileHelper.readUTF8File(t)
        if content is None:
            continue
        try:
            base_url=content.splitlines()[0]
        except IndexError or AttributeError:
            continue

        document=html2words(content, base_url, encoding='utf-8', supervisior=supervisior)
        if document is None:
            os.remove(t)
            continue

        pure_text='\t'.join([document['title'], document['meta_descrip'], document['para']])

        predicted_y, precision=supervisior.predict(pure_text)

        tree_file=os.path.join(tree, f)
        json_file=os.path.join(json_dir, f.split('.')[0] + '.json')

        save(document, json_file, tree_file)

        key=f.split('.')[0]
        urlitems_file=os.path.join(url, key)

        cPickle.dump(document['url_items'], open(urlitems_file, 'wb'), -1)

        item=gen_url_item(base_url, document['title'], predicted_y, precision)
        url_sample[key]=item
        document_class[key]=predicted_y

        # with open(url_sample_file, 'a') as fu:
        #     fu.write('%s\t%s\t%s\t%s\t%.3f\n' % (
        #     key, encode(item['url']), encode(item['anchor_text']), item['label'], item['interestness']))


        finput.writelines(f + '\n')
    finput.close()

    # cPickle.dump(url_sample, open(url_sample_pickle, 'wb'), -1)
    json.dump(document_class, open(dcoument_class_file, 'w'))


def save_response(name, response, count, supervisor1, supervisior2):
    save_dir = '/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/%s/' %name
    original = save_dir+'original/'
    json_dir = save_dir +'json/'
    temp = save_dir + 'temp/'
    mkdirs(save_dir,original, json_dir, temp)


    html_file = original + str(count)+'.html'
    json_file = json_dir + str(count)+'.json'
    url_sample_1 = open(temp + 'url.sample1.txt', 'a')
    url_sample_2 = open(temp + 'url.sample2.txt', 'a')

    if response is None:
        return


    link = response.meta.get('link', '')
    url = response.url
    body = response.body
    encoding = response.encoding

    content = url +'\n'+body
    try:
        with open(html_file, 'w') as fw:
            # content = content.decode(encoding).encode('utf-8', 'ignore')
            content = EncodeHelper.get_unicode(content, encoding)
            if content is None:
                return
            content = content.encode('utf-8')
            fw.write(content)
    except UnicodeError:
        return

    document = html2words(content, url, 'utf-8')
    if document is None:
        return

    pure_text='\t'.join([document['title'], document['meta_descrip']*3, document['para']])
    predicted_y, precision = supervisor1.predict(pure_text)
    #count:parent_url:url:anchor_text:label:precision
    url_sample_1.write('%d\t%s\t%s\t%s\t%s\t%.3f\n' %(count,
                                                      link['parent_url'],
                                                      link['url'],
                                                      link['anchor_text'],
                                                      str(predicted_y),
                                                      precision))
    predicted_y, precision = supervisior2.predict(pure_text)
    #count:parent_url:url:anchor_text:label:precision
    url_sample_2.write('%d\t%s\t%s\t%s\t%s\t%.3f\n' %(count,
                                                      link['parent_url'],
                                                      link['url'],
                                                      link['anchor_text'],
                                                      str(predicted_y),
                                                      precision))
    url_sample_1.close()
    url_sample_2.close()
    save(document, json_file)


def create_train_url(name):
    # src_dir = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' % name
    #
    # supervisior=MultiClassifier('test-6-topic')
    # supervisior.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')
    #
    # home = src_dir
    # yule = home+'yule/'
    # sport = home +'sport/'
    # finance = home +'finance/'
    # junshi = home + 'junshi/'
    #
    #
    # url_train_path=os.path.join(src_dir, 'url_train.txt')
    #
    # with open(url_train_path, 'w') as fw:
    #     for original in [yule, sport, finance, junshi]:
    src_dir = '/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/'
    supervisior=MultiClassifier('test-6-topic')
    supervisior.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')
    url_train_path=os.path.join(src_dir, 'url_train.txt')
    with open(url_train_path, 'w') as fw:
        for dir in ['qq/', 'sina/', 'ifeng/', 'sohu/']:
            original = src_dir + dir + 'original/'
            for f in os.listdir(original):
                print '========================================'
                print 'parse file: %s ....' % f
                t=os.path.join(original, f)

                content=FileHelper.readUTF8File(t)
                try:
                    if content is None:
                        continue

                    base_url=content.splitlines()[0]
                except IndexError:
                    continue

                for link in get_link_word_by_pair(content, base_url, supervisior, 'utf-8'):
                    try:
                        fw.write('%s\t%s\t%s\t%s\t%.3f\n' % (
                        link['parent_url'], link['url'], link['anchor_text'], str(link['label']),
                        link['interestness']))
                    except UnicodeDecodeError:
                        continue



#================================================================================
# 下面的程序用于测试wish-ce算法在进行网页分类时的有效性
#get_page_para:分别使用cetd, wish-ce, remove-tag三种方法从网页中抽取正文文本
#predict:分别利用三种方法抽取得到的文本进行分类预测
#Result:比较三种方法的实验结果
def mkdirs(*args):
    for dir in args:
        if not os.path.exists(dir):
            os.mkdir(dir)


def get_page_para(name):
    home = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' %name

    cetd = home + 'plain-text/cetd/'
    wish_ce = home + 'plain-text/wish-ce/'
    remove_tag_ = home + 'plain-text/remove-tag/'

    mkdirs(cetd, wish_ce, remove_tag_)

    history = home+'history.txt'
    visited_files = get_history(history)
    finput = open(history, 'w')


    yule = home+'yule/'
    sport = home +'sport/'
    finance = home +'finance/'
    junshi = home + 'junshi/'


    for dir in [yule, sport, finance, junshi]:
        for f in os.listdir(dir):
            if f in visited_files:
                continue
            print '========================================'
            print 'parse file: %s ....' % f

            t = dir + f
            content=FileHelper.readUTF8File(t)
            if content is None:
                continue
            try:
                base_url=content.splitlines()[0]
            except IndexError or AttributeError:
                continue


            docstring=content.lower()
            doctree=HtmlHelper.create_doc(docstring, 'utf-8')

            if doctree is None: return None

            cetd_doc = copy.deepcopy(doctree)

            try:
                article_c = get_aricle_cetd(cetd_doc)


                doctree=HtmlHelper.pre_process_domtree(doctree)

                article_w, title = HtmlHelper.get_article(doctree)
                article_w = ' '.join([title, article_w])

                article_r = remove_tags(docstring)

                finput.writelines(f + '\n')
            except:
                continue
            f = f.split('.')[0] + '.txt'
            FileHelper.WriteInUTF8(cetd+f, article_c)
            FileHelper.WriteInUTF8(wish_ce+f, article_w)
            FileHelper.WriteInUTF8(remove_tag_+f, article_r)

    finput.close()


def predict(name):
    home = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' %name

    temp = home+'temp/'
    mkdirs(temp)

    supervisior=MultiClassifier('test-3-topic')
    supervisior.load(path='/mnt/UbutunShare/TopicalCrawl/TopicalCrawl/classifier')

    cetd = home + 'plain-text/cetd/'
    wish_ce = home + 'plain-text/wish-ce/'
    remove_tag_ = home + 'plain-text/remove-tag/'

    history = temp+'history.txt'
    visited_files = get_history(history)
    finput = open(history, 'w')


    document_class = {}
    for dir in [cetd, wish_ce, remove_tag_]:
        class_ = '1'
        if 'cetd' in dir:class_ = '0'
        elif 'wish' in dir:class_ = '1'
        else: class_ = '2'
        print dir
        print '========================'
        document_class[class_] = {}

        tmpfile = open(temp+class_+'.txt', 'w')
        for f in os.listdir(dir):
            if f in visited_files:
                continue
            print '==============================='
            print f

            t = dir + f
            content=FileHelper.readUTF8File(t)
            if content is None:
                continue

            predicted_y, precision=supervisior.predict(content)
            tmpfile.write('%s\t%s\n' %(predicted_y, f))

            document_class[class_][f] = predicted_y
            finput.writelines(f + '\n')
        tmpfile.close()
    finput.close()
    json.dump(document_class, open(temp+'test_result.txt', 'w'))

def del_dic(dic, key):
    if key in dic:
        del dic[key]



def result(name):
    from TopicalCrawl.classifier.base import TestResult
    import numpy
    home = '/mnt/UbutunShare/graduate/DataSet/PageClassification/%s/' %name

    temp = home+'temp/'
    document_labels = json.load(open(temp+'test_result.txt'))



    exp=lambda x:x.split('.')[0]
    em = lambda x:x.encode('utf-8')

    id2file = {}

    true_y = {}
    with open(temp+'true_y.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label, _file = line.split('\t')
            true_y[exp(_file)] = label
    file2id = {_file:_id for _id, _file in enumerate(true_y)}

    cetd, wish, rt = {}, {}, {}

    cetd = document_labels['0']
    wish = document_labels['1']
    rt = document_labels['2']

    cetd = {em(exp(_file)):_label.encode('utf-8') for _file, _label in cetd.items()}
    wish = {em(exp(_file)):_label.encode('utf-8') for _file, _label in wish.items()}
    rt = {em(exp(_file)):_label.encode('utf-8') for _file, _label in rt.items()}

    cetd_array = numpy.zeros(len(true_y))
    wish_array = numpy.zeros(len(true_y))
    rt_array = numpy.zeros(len(true_y))
    _true_y =  numpy.zeros(len(true_y))

    for _file in true_y:
        _id = file2id[_file]
        try:
            cetd_array[_id] = cetd[_file]
            wish_array[_id] = wish[_file]
            rt_array[_id] = rt[_file]
            _true_y[_id] = true_y[_file]
        except KeyError:
            continue


    cetd_result = TestResult(_true_y, cetd_array, None)
    wish_result = TestResult(_true_y, wish_array, None)
    rt_result = TestResult(_true_y, rt_array, None)

    print '============cetd=============='
    print cetd_result.accuracy_labels
    print cetd_result.recall_labels
    print cetd_result.accuracy_score
    cetd_result.show_result()
    print '============wish=============='
    print wish_result.accuracy_labels
    print wish_result.recall_labels
    print wish_result.accuracy_score
    wish_result.show_result()
    print '============rt=============='
    print rt_result.accuracy_labels
    print rt_result.recall_labels
    print rt_result.accuracy_score
    rt_result.show_result()








if __name__ == '__main__':
    # create_train_samples('sohu')
    # create_train_url('Test5')
    name = 'Test5'
    get_page_para(name)
    # predict(name)
    #
    # result(name)













