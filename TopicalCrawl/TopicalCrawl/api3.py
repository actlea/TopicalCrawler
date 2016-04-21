#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: api.py
@time: 16-3-3 下午4:12
"""

import collections
import copy
import lxml
from lxml.html.clean import Cleaner

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

HTML_CLEANER = Cleaner(scripts=True, javascript=True, comments=True,
                       style=False, links=True, meta=False, add_nofollow=False,
                       page_structure=False, processing_instructions=True,
                       embedded=True, frames=True, forms=True,
                       annoying_tags=True,
                       # remove_tags=["i", "em", "b" ], #"strong", "span"
                       kill_tags=("iframe", "figure", "noscript"), #noscript
                       remove_unknown_tags=True, safe_attrs_only=False)

from util import ElementHelper, StringHelper, FileHelper, EncodeHelper
from config import *

#===========================================================
#base function
def normalize_word(content):
    return StringHelper.normalize_word(content)

def replaceAll(pattern, src, tar):
    return StringHelper.replaceAll(pattern, src, tar)

def split_words(content):
    content = StringHelper.chinese_segment(content)
    content = content.lower()
    content = content.split()
    import string
    punctuation = [unicode(p) for p in string.punctuation]
    chinese_punctuation = u',?、。“”《》！，：；？＞＜＠＃＄％︿＆＊＋～｜［］｛｝-'
    chinese_punctuation = StringHelper.chinese_segment(chinese_punctuation).split()
    punctuation.extend(chinese_punctuation)

    content=[c for c in content if c not in punctuation]
    return content



#==========================================================
def set_element_attribute(element, x, y):
    """ x is level of element, y is PreOrder number
    """
    #set_element_coordinate
    element.set(px,str(x))
    element.set(py, str(y[0]))
    y[0] += 1

    #set element record, it mark whether an element is ancestor node of a record
    element.set(kg_record_mark, '0')

    #kg_text_mark mark whether an element belong to main text part
    element.set(kg_text_mark, '0')

    #set element .text and .tail
    element.tail = normalize_word(element.tail)
    element.text = normalize_word(element.text)

    if len(element)==0 and len(element.tail)==0 and len(element.text)==0:
        ElementHelper.remove_element(element)

    for child in element:
        set_element_attribute(child, x+1, y)

def is_set_attribute_valid(element):
    try:
        set_element_attribute(element, 1, [1])
        return True
    except:
        return False

def is_elements_belong_para(elements):
    for node in elements:
        if node.get(kg_mark)!='0':
            return True
    return False


#=========================================================================
# find html title in body
def longest_common_length(target, pattern):
    target_list = split_words(target)
    patterns_list = split_words(pattern)

    lengths = [[0 for j in range(len(patterns_list)+1)] for i in range(len(target_list)+1)]

    for i, x in enumerate(target_list):
        for j, y in enumerate(patterns_list):
            # if x == y:  ToDo: change 2016-03-07
            if x in y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

    result = []
    x, y = len(target_list), len(patterns_list)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x-1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y-1]:
            y -= 1
        else:
            # assert target_list[x-1][0] == patterns_list[y-1][0]
            assert target_list[x-1] in patterns_list[y-1]
            # result = a[x-1] + result
            result.append(target_list[x-1])
            x -= 1
            y -= 1
    return len(result[::-1])

def is_possible_title_tag(element):
    if element.tag not in TITLE_TAG:
        return False
    if len(element)>1:
        return False
    if ElementHelper.is_element_content_none(element):
        return False
    return True

def get_title_util(body, title_text):
    if len(title_text) < 2:
        return None
    #1.计算节点文本与title_text的longest commen lenght
    candidate_nodes = []
    for tag in TITLE_TAG:
        nodes = ElementHelper.get_element_by_tag(body, tag)
        if nodes is None or len(nodes)<1: continue

        nodes = [node for node in nodes if is_possible_title_tag(node)]
        candidate_nodes.extend(nodes)

    mapping = {}
    for node in candidate_nodes:
        node_text = ElementHelper.element_text_content(node)
        # if len(node_text)==0 or len(node_text)>len(title_text): continue  #

        if len(node_text)==0: continue

        llength = longest_common_length(node_text, title_text)
        if llength >= 1:
            mapping[node] = llength

    if len(mapping)==0: return None
    #2.选择长度最大的作为title节点，如果存在多个最大的，选择最靠前的作为
    #title节点
    sorted_nodes = [node for node, _ in sorted(mapping.items(), key=lambda x:x[1], reverse=True)]
    max_len = mapping[sorted_nodes[0]]

    candidates = [node for node in sorted_nodes if mapping[node]==max_len]
    if len(candidate_nodes)==1:
        return sorted_nodes[0]
    else:
        candidates.sort(cmp=lambda x,y: ElementHelper.get_element_preorder_num(x)- ElementHelper.get_element_preorder_num(y), reverse=False)
        return candidates[0]

#=============================================================
# filter non need cluster
def cluster_text_number(cluster):
    numbers = [ElementHelper.get_element_content_number(node) for node in cluster ]
    return sum(numbers)


def is_cluster_all_links(cluster):
    """ if all tags which contain links are <a> tag, then return True
    For example:
        <a> link </a>
        OR
        <li> <a> link </a> </li>
    """
    all_nodes_contain_text = []
    for node in cluster:
        children = ElementHelper.get_children(node)
        nodes_contain_text = [node for node in children if not ElementHelper.is_element_text_none(node)
            and node.tag not in ('em','strong','span','i','b')]
        all_nodes_contain_text.extend(nodes_contain_text)

    link_nodes = [node for node in all_nodes_contain_text if node.tag=='a' or node.getparent().tag=='a']
    other_nodes = [node for node in all_nodes_contain_text if node.tag!='a' and node.getparent().tag != 'a']

    link_nodes_text_number = cluster_text_number(link_nodes)
    other_nodes_text_number = cluster_text_number(other_nodes)

    if len(other_nodes)==0 or other_nodes_text_number==0:
        return True

    if 1.0 *link_nodes_text_number/other_nodes_text_number>2.0:
        return True

    return False

def is_cluster_possible_main_text(cluster):
    for node in cluster:
        mark = node.get(kg_mark)
        if mark=='2' or mark=='1':
            return True
    return False

def is_link_node_with_text(element):
    """ if element.text_content not None and element=='a', return True
    """
    if not ElementHelper.is_element_content_none(element) and element.tag=='a':
        return True
    return False

def is_cluster_contain_user_comments(cluster):
    """ identify whether element or its children contain comment content, only consider <a> tag
    1.each node in cluster, at least has 3 children
    2.there is at least one <a> tag has same text
    """
    # can not identify
    if len(cluster) < 2: return False

    text_number_mapping = {}

    #at least have three children contain text
    for node in cluster:
        children = ElementHelper.get_children(node)
        link_nodes_contain_text = [n for n in children if is_link_node_with_text(n)]
        if len(link_nodes_contain_text)<3: return False

        for n in link_nodes_contain_text:
            text = ElementHelper.element_text_content(n)
            if text in text_number_mapping:
                text_number_mapping[text] += 1
            else:
                text_number_mapping[text] = 1
    #去除标点符号，出数字，空的文本
    for text in text_number_mapping:
        if len(text)==0 or StringHelper.is_digits(text) :
            del text_number_mapping[text]

    text_number = text_number_mapping.values()

    # FOR TEST
    # for text, number in node_text_mapping.items():
    #     print text,':', number

    text_number_counter = collections.Counter(text_number).most_common()

    for number, counter in text_number_counter:
        if number > 1 and number==len(cluster) and counter>=1: #ToDo 2016/03/08  old:counter>=2 --> new:counter>=1
            print 'find comment!'
            return True
    return False


def is_valid_cluster(nodes):
    if not is_cluster_contain_user_comments(nodes) \
        and not is_cluster_all_links(nodes) and len(nodes)>1:
        return True
    return False

def filter_cluster(clusters):
    candidate_clusters = {}
    for cluster, nodes in clusters.items():
        if not is_cluster_contain_user_comments(nodes) \
            and not is_cluster_all_links(nodes) : # and len(nodes)>1 ToDo:2016-0308
            candidate_clusters[cluster] = nodes

    return candidate_clusters

def get_biggest_cluster(clusters):
    maxNumber = 0
    maxCluster = None
    for cluster, nodes in clusters.items():
        text_number = cluster_text_number(nodes)
        if text_number > maxNumber:
            maxNumber = text_number
            maxCluster = cluster

    return maxCluster

def set_text_mark(element, s, t):
    for child in element:
        set_text_mark(child, s, t)

    pos = ElementHelper.get_element_preorder_num(element)
    if pos >=s and pos<=t:
        element.set(kg_text_mark, '1')
        parent = element.getparent()
        while parent.tag != 'html':
            parent.set(kg_text_mark, '2')
            parent = parent.getparent()

def remove_nontext_element(element):
    if element.get(kg_text_mark, '0') == '0':
        ElementHelper.remove_element(element)

    for child in element:
        remove_nontext_element(child)

def get_article_wish(clusters, doctree, title_node=None, debug = False):
    #filter
    clusters = filter_cluster(clusters)

    if len(clusters) == 0:
        return ElementHelper.element_text_content(title_node)

    maxCluster = get_biggest_cluster(clusters)

    nodes = clusters[maxCluster]

    m = {}
    m[maxCluster] = nodes
    filter_cluster(m)

    #============================================
    if debug:
    # for test
        print 'maxCluster: %s' %maxCluster
        for n in nodes:
            print ElementHelper.get_xpath_by_element(n, doctree)

    allnodes = []

    for node in nodes:
        children = ElementHelper.get_children(node)
        allnodes.extend(children)

    allnodes.sort(lambda p,q:ElementHelper.get_element_preorder_num(p)-ElementHelper.get_element_preorder_num(q))
    s = ElementHelper.get_element_preorder_num(allnodes[0])
    t = ElementHelper.get_element_preorder_num(allnodes[-1])
    title_text = ''
    if title_node is not None:
        index = ElementHelper.get_element_preorder_num(title_node)
        if index < s:
            s = index
            title_text = ElementHelper.element_text_content(title_node)

    body = ElementHelper.get_body(doctree)

    set_text_mark(body, s, t)

    remove_nontext_element(body)

    # ElementHelper.print_element(body)

    return ElementHelper.element_text_content_list(body), title_text



#==========================================================
# WISH to get html segments
class WISH:
    @classmethod
    def similar_check(cls, nodeA, nodeB):
        if nodeA.tag != nodeB.tag:
            return False
        #compare distinct nodes
        dnodesA = ElementHelper.get_children(nodeA)
        dnodesB = ElementHelper.get_children(nodeB)

        #dA is node_levels_mapping, rA is level_nodes_mapping
        dA, dB, rA, rB = {}, {}, {}, {}
        for node in dnodesA:
            #ignore <a> tag as distinct tag
            if node.tag == 'a': continue

            dA.setdefault(node.tag, []).append(int(node.get(px)))
            rA.setdefault(int(node.get(px)), []).append(node.tag)
        for node in dnodesB:
            if node.tag == 'a': continue

            dB.setdefault(node.tag, []).append(int(node.get(px)))
            rB.setdefault(int(node.get(px)), []).append(node.tag)

        if abs(len(dA)-len(dB))>1 or abs(len(rA)-len(rB))>1:
            return False

        #check distinct tag is same？
        for tag in dA:
            if tag not in ('em', 'b', 'br','i') and tag not in dB:
                return False

        sumA = sum([len(StringHelper.unique(rA[A])) for A in rA])
        sumB = sum([len(StringHelper.unique(rB[B])) for B in rB])
        if abs(sumA-sumB)>1:
            return False
        return True

    @classmethod
    def bfs_tree(cls, doctree):
        all_level_nodes = {}
        body = ElementHelper.get_body(doctree)

        level = int(body.get(px))
        thislevel = [body]
        while thislevel:
            nextlevel = list()
            all_level_nodes[level] = thislevel
            for node in thislevel:
                nextlevel.extend([child for child in node if len(node)>0])
            thislevel = nextlevel
            level += 1
        return all_level_nodes

    @classmethod
    def is_node_or_ancestor_record(cls, element):
        if element.get(kg_record_mark)=='1':
            return True
        parent = element.getparent()
        while parent.tag != 'html':
            if parent.get(kg_record_mark)=='1':
                return True
            parent = parent.getparent()
        return False

    @classmethod
    def is_node_record(cls, element):
        return element.get(kg_record_mark) == '1'

    @classmethod
    def is_node_or_children_record(cls, element):
        children = ElementHelper.get_children(element)

        marks = [child.get(kg_record_mark) for child in children]
        unique_marks = StringHelper.unique(marks)
        if len(unique_marks)==2:
            return True
        return False

    @classmethod
    def correct_record_mark(cls, element):
        """ if element's children are squence records, like: [1] [2][4] [5], then mark [3] as record
        """
        if len(element) < 1:
            return

        node_index_mapping, index_node_mapping = {}, {}
        cnt = 1

        for child in element:
            node_index_mapping[child] = cnt
            index_node_mapping[cnt] = child

            cnt += 1

        for i in range(2, cnt-1):
            left_mark = index_node_mapping[i-1].get(kg_record_mark)
            right_mark = index_node_mapping[i+1].get(kg_record_mark)
            mark = index_node_mapping[i].get(kg_record_mark)

            if mark=='0' and left_mark=='1' and right_mark=='1' \
                and not cls.is_node_or_children_record(index_node_mapping[i]):
                index_node_mapping[i].set(kg_record_mark, '1')

    @classmethod
    def find_first_sibling_record_node(cls, element, doctree):
        parent = element.getparent()
        if len(parent)<2:
            return element

        element_xpath = ElementHelper.get_xpath_by_element(element, doctree)
        # print 'xpath: %s' %element_xpath
        element_last_index = StringHelper.get_digits(element_xpath.split('/')[-1])

        if element_last_index < 2:
            return element

        index = element_last_index - 2
        # print 'parent length:%d' %len(parent)
        while index >= 0:
            # print index
            if parent[index].get(kg_record_mark) == '1':
                index -= 1
            else:
                break
        return parent[index+1]

    @classmethod
    def get_clustered_records(cls, doctree):
        #get level_nodes_mapping
        all_level_nodes = cls.bfs_tree(doctree)

        root = ElementHelper.get_root(doctree)
        body = ElementHelper.get_body(doctree)

        #get max level and min level
        upper_bound = int(ElementHelper.get_element_depth(root))+1
        low_bound = int(body.get(px))+1

        for level in range(low_bound, upper_bound):

            level_nodes = all_level_nodes[level]
            #if parent is record node, then do not consider its children
            level_nodes = [node for node in level_nodes if not cls.is_node_or_ancestor_record(node)]

            for j in range(1,len(level_nodes)-1):
                left_node = level_nodes[j-1]
                #横向比较
                right_bound = min(len(level_nodes), j+5)
                right_nodes = level_nodes[j:right_bound]

                #纵向比较
                down_nodes = right_nodes[0]
                right_nodes.extend(down_nodes)

                for right_node in right_nodes:
                    if cls.similar_check(left_node, right_node):
                        left_node.set(kg_record_mark,'1')
                        right_node.set(kg_record_mark, '1')
                        break

        record_groups = cls.merger_sibling_record_node(doctree)
        return record_groups

    @classmethod
    def merger_sibling_record_node(cls, doctree):
        node_record_mapping = {}

        body = ElementHelper.get_body(doctree)
        thislevel = []
        thislevel.extend(body)
        while thislevel:
            nextlevel = list()
            for node in thislevel:
                # correct nodes which
                cls.correct_record_mark(node)

                if cls.is_node_or_ancestor_record(node):
                    first_record_sibling = cls.find_first_sibling_record_node(node, doctree)
                    node_record_mapping.setdefault(first_record_sibling, []).append(node)

            for node in thislevel:
                if len(node) > 0:
                    nextlevel.extend([child for child in node if not cls.is_node_or_ancestor_record(node)])
            thislevel = nextlevel

        return node_record_mapping

    @classmethod
    def print_cluster_record(cls, clusters, doctree):
        ElementHelper.print_element(doctree)
        for cluster in clusters:
            if len(cluster)>1:
                print '===='*10
                nodes = clusters[cluster]
                for node in nodes:
                    print ElementHelper.get_xpath_by_element(node, doctree)

#=======================================================
# extract page main context


#=======================================================
class HtmlHelper(object):
    @staticmethod
    def get_cleaned_html(content):
        """ remove scripts, style and non-needed tag
        """
        tmp = copy.deepcopy(content)
        # try:
        #     content = HTML_CLEANER.clean_html(content)
        # except:
        #     content = tmp
        content = replaceAll(r'(?s)<!--.*?-->', '', content)
        content = replaceAll(r'(?s)<script.*?>.*?</script>', '', content)
        content = replaceAll(r'(?s)<noscript.*?>.*?</noscript>', '', content)
        content = replaceAll(r'(?s)<SCRIPT.*?>.*?</SCRIPT>', '', content)
        content = replaceAll(r'(?s)<style.*?>.*?</style>', '', content)
        # content = replaceAll(r'(?s)<form.*?>.*?</form>', '', content)
        content = replaceAll(r'(?s)<fieldset.*?>.*?</fieldset>', '', content)
        content = replaceAll(r'[\r,\n]\s*\n+', '', content) #去除空行
        return content

    @staticmethod
    def is_valid_html(html, encoding=None):
        pass

    @staticmethod
    def create_doc(html, encoding=None):
        if isinstance(html, unicode):
            html = html.encode('utf-8')
        elif not encoding=='utf-8':
            html = EncodeHelper.get_unicode(html, encoding).encode('utf-8')

        content = HtmlHelper.get_cleaned_html(html)
        doctree = lxml.html.parse(BytesIO(content),
                                  lxml.html.HTMLParser(encoding='utf-8',
                                                       remove_blank_text=True))
        return doctree


    @staticmethod
    def pre_process_domtree(doctree):
        if doctree is not None:
            root = ElementHelper.get_root(doctree)
            if is_set_attribute_valid(root):
                return doctree
            else:
                return None
        else:
            return None

    @staticmethod
    def get_title(doctree):
        try:
            return doctree.find('.//title')
        except AttributeError:
            return None

    @staticmethod
    def get_article_title_element(doctree):
        body = ElementHelper.get_body(doctree)

        title_node = HtmlHelper.get_title(doctree)

        if title_node is None:
            return None

        title_text = title_node.text
        title = get_title_util(body, title_text)
        if title is None: return None
        return title

    @staticmethod
    def get_article(doctree, debug = False):
        w = WISH()

        title = HtmlHelper.get_article_title_element(doctree)

        clusters = w.get_clustered_records(doctree)

        if debug:
            w.print_cluster_record(clusters, doctree)

        articles = get_article_wish(clusters, doctree, title, debug)
        return articles


html1 = """
<div kg_x="1">
    <div kg_x="2"> this is example
        <a kg_x="3"> links </a>
    </div>
    <div kg_x="2"> this is example</div>
    <div kg_x="2"> this is example</div>
</div>
"""

html2 = """
<div kg_x="1">
    <div kg_x="2"> this is example
        <div kg_x="3"> this is example
            <em kg_x="4">dsda</em>
        </div>
    </div>
    <div kg_x="2"> this is example</div>
</div>
"""




if __name__ == '__main__':
    import lxml.etree as etree
    doc1 = etree.fromstring(html1)
    doc2 = etree.fromstring(html2)

    root1 = ElementHelper.get_root(doc1)
    root2 = ElementHelper.get_root(doc2)
    w = WISH()
    # print w.similar_check(root1, root2)
    dir = 'D:/OtherWork/CETD_DATA/BBC/original/0.htm'
    dir = '/mnt/UbutunShare/Work/CETD_DATA/BBC/original/71.htm'
    dir = '/mnt/UbutunShare/Work/CETD_DATA/Final/original/60.html'
    # dir = '/mnt/UbutunShare/Work/CETD_DATA/wiki/original/1.htm'
    from util import FileHelper
    content = FileHelper.readUTF8File(dir)
    doc = HtmlHelper.create_doc(content, 'utf-8')
    doc = HtmlHelper.pre_process_domtree(doc)
    article,title = HtmlHelper.get_article(doc, False)
    ElementHelper.print_element(doc)
    print article.encode('utf-8')







