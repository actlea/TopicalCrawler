#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: util.py
@time: 16-3-3 上午9:21
"""
import logging
import codecs
import chardet
import time
import os
import re
from lxml.etree import ParseError,XPathEvalError
from lxml import etree
import random

from config import *
from settings import USER_AGENTS

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



class EncodeHelper(object):
    @staticmethod
    def get_unicode(content, encoding=None):
        if isinstance(content, unicode):
            return content
        tmp = content

        if encoding is not None:
            try:
                content = tmp.decode(encoding)
                return content
            except UnicodeDecodeError:
                pass

        encode_dic = {1:'utf-8', 2:'gb2312', 3:'gbk'}
        for i in range(1,4):
            try:
                content = tmp.decode(encode_dic[i])
                return content
            except UnicodeDecodeError:
                continue
        return None



            # if encoding is None:
            #     encoding = chardet.detect(tmp)['encoding']
            #     try:
            #         content = tmp.decode(encoding)
            #     except UnicodeDecodeError:
            #         return None



class FileHelper(object):
    @staticmethod
    def loadResourceFile(filename):
        if not os.path.isabs(filename):
            dirpath = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(dirpath, 'resources', filename)
        else:
            path = filename
        try:
            f = codecs.open(path, 'r', 'utf-8')
            content = f.read()
            f.close()
            return content
        except UnicodeDecodeError:
            f = open(path, 'r')
            content = f.read()
            return EncodeHelper.get_unicode(content)
        except IOError:
            raise IOError("Couldn't open file %s" % path)

    @staticmethod
    def readUTF8File(filename):
        try:
            inFile = codecs.open(filename, 'r', 'utf-8')
            content = inFile.read()
            inFile.close()
            return content
        except UnicodeDecodeError:
            f = open(filename, 'r')
            content = f.read()
            content = EncodeHelper.get_unicode(content)
            if content is None:
                return None
            content = content.encode('utf-8', 'ignore')
            f.close()
            return content
        except IOError:
            raise IOError("Couldn't open file %s" % filename)

    @staticmethod
    def readFile(filePath):
        try:
            with open(filePath, 'r') as f:
                content = f.read()
            return content
        except IOError:
            raise IOError("Couldn't open file %s" % filePath)

    @staticmethod
    def WriteInUTF8(filename, content, mode='w', encoding='utf-8'):
        try:
            inFile = codecs.open(filename, mode, 'utf-8')
            inFile.write(content)
            inFile.close()
        except UnicodeDecodeError:
            f = open(filename, mode)
            content = EncodeHelper.get_unicode(content, encoding)
            content = content.encode('utf-8', 'ignore')
            f.write(content)
            f.close()
        except IOError:
            raise IOError("Couldn't open file %s" % filename)





    @staticmethod
    def WriteListText(list_texts=[],outputFileName=''):
        contents = '\n'.join(list_texts)
        FileHelper.WriteInUTF8(outputFileName,contents )



class StringHelper(object):
    # @staticmethod
    # def normalize_word(word):
    #     """remove empty lins, empty tab, multi blank
    #     """
    #     if word is None:
    #         return u''
    #     if not isinstance(word, unicode):
    #         word = word.decode('utf-8', 'ignore')
    #
    #     word = re.sub(r'\t', ' ', word)
    #     word = re.sub(r'[\r,\n]\s*[\n,\t]+','\n',word) #去除空行
    #     word =  u' '.join([x.strip() for x in word.split(' ') if len(x.strip())!=0])
    #     if isinstance(word, unicode):
    #         return word
    #     else:
    #         return word.decode('utf-8')

    @staticmethod
    def stripTags(pageContents):

        startLoc = pageContents.find("<html")

        pageContents = pageContents[startLoc:]

        inside = 0
        text = ''
        for char in pageContents:
            if char == '<':
                inside = 1
            elif (inside == 1 and char == '>'):
                inside = 0
            elif inside == 1:
                continue
            else:
                text += char
        return text

    @staticmethod
    def normalize_word(word):
        """remove empty lins, empty tab, multi blank
        """
        if word is None:
            return ''
        try:
            if not isinstance(word, unicode):
                word = word.decode('utf-8', 'ignore')

            word = re.sub(r'\t', ' ', word)
            # word = re.sub(r'[\r,\n]\s','',word) #去除行 #ToDo 2016-04-20
            word = re.sub(r'[\r,\n]\s*[\n,\t]+', '\n', word)  # 去除空行
            word =  u' '.join([x.strip() for x in word.split(' ') if len(x.strip())!=0])

            return word
        except:
            return ''


    @staticmethod
    def chinese_segment(sentences):
        """ segement chinese string, eg. '我来到北京'->'我 来 到 北 京'
        """
        if not isinstance(sentences, unicode):
            try:
                sentences = sentences.decode('utf-8')
            except UnicodeDecodeError:
                encoding = chardet.detect(sentences)['encoding']
                sentences = sentences.decode(encoding)
        result= u''
        for s in sentences:
            # if s>=u'\u4e00' and s<= u'\u9fff':
            if s>=u'\u2E80' and s<= u'\u9fff':
                result += u' ' + s + u' '
            else:
                result += s
        return StringHelper.normalize_word(result)

    @staticmethod
    def simplify_xpath(xpath, index=None):
        """ `/html/body/div[2]/div[1]` -> `/html/body/div/div`
        if index=-1 ,`/html/body/div[2]/div[1]` -> `/html/body/div[2]/div`
        """
        if index is None:
            return re.sub('\[\d+\]', '', xpath)
        else:
            pre_xpath = '/'.join(xpath.split('/')[0:index])
            return '/'.join([re.sub('\[\d+\]', '', pre_xpath),xpath.split('/')[index]])

    @staticmethod
    def get_digits(str):
        """ `[23]` -> 23
        """
        try:
            return int(re.search(r'\[\d+\]', str).group(0).strip('[]'))
        except AttributeError:
            return 0

    @staticmethod
    def is_digits(content):
        content = StringHelper.normalize_word(content)
        for i in content:
            if i not in '0123456789.':
                return False
        return True

    @staticmethod
    def index_of(target, patterm, _from=0):
        """ find pattern in target from _from index
        """
        try:
            index = target.find(patterm, _from)
        except TypeError:
            index = -1
        return  index

    @staticmethod
    def is_prefix(target, src):
        """test if target are prefix of src
        """
        if len(target)>len(src):
            return False
        for i in range(len(target)):
            if target[i]!=src[i]:
                return False
        return True

    @staticmethod
    def unique(_list):
        return list(set(_list))

    @staticmethod
    def replaceAll(pattern, src, tar):
        return re.sub(pattern, src, tar)


class BasicConfig(object):
    def __init__(self):
        pass


class ElementHelper(object):
    @staticmethod
    def remove_element(node):
        node.getparent().remove(node)

    @staticmethod
    def element_text_content(element):
        """ get all text content between start tag and end tag
        """
        try:
            content = element.text_content()
            return StringHelper.normalize_word(content)
        except AttributeError or ValueError:
            return u''

    @staticmethod
    def element_text_tail(element):
        if element is None:
            return u''
            #获取<p>text</p>中的text
        if not ElementHelper.is_element_text_none(element):
            result = element.text
        else:
            result = u''
            #获取<p><a></a>text</p>中的text
        for child in element:
            if not ElementHelper.is_element_tail_none(child):
                result += child.tail
        return result

    @staticmethod
    def element_text_content_list(element):
        try:
            content = element.text
            if content is None:
                content=u''

            for child in element:
                content += '\n' + ElementHelper.element_text_content_list(child)
                if child.tail is not None:
                    content += u' '+child.tail
            return StringHelper.normalize_word(content)
        except AttributeError:
            return u''

    @staticmethod
    def get_elements_by_tagnames(element, tagnames):
        """ return all element which element.tag = tagname
        """
        if element is None:
            yield None
        for tag_name in tagnames:
            for e in element.findall('.//%s' %tag_name):
                yield e


    @staticmethod
    def get_element_by_tag(element, tag):
        try:
            return element.findall('.//%s' %tag)
        except AttributeError:
            return None


    @staticmethod
    def get_element_by_attr(element, attr, value=None):
        """ return all elements which has [@attr=value]
        """
        if value is not None:
            return element.findall('.//*[@%s="%s"]' %(attr, value))
        else:
            return element.findall('.//*[@%s]' %attr)


    @staticmethod
    def get_element_by_tag_attr(element, tag, attr, value):
        """return all elements which `tag[@attr=value]`
        """
        return element.findall('.//%s[@%s="%s"]' %(tag, attr, value))


    @staticmethod
    def get_element_by_xpath(xpath, doctree):
        """ return element by its xpath
        """
        try:
            return doctree.xpath(xpath)[0]
        except (ParseError or XPathEvalError or IndexError):
            return None


    @staticmethod
    def get_xpath_by_element(element, doctree):
        try:
            return doctree.getpath(element)
        except TypeError or ValueError:
            return None


    @staticmethod
    def get_attribute(element,attr=None):
        """ get all elements which has `attr`
        """
        if attr:
            return element.attrib.get(attr, None)
        return None

    @staticmethod
    def get_body(doctree):
        try:
            body = doctree.find('.//body')
            return body
        except AttributeError or TypeError:
            return None

    @staticmethod
    def get_root(doctree):
        try:
            root = doctree.getroot()
        except AttributeError:
            root = doctree.getroottree().getroot()
        return root

    @staticmethod
    def createElement(tag='p', text=None, tail=None):
        node = etree.Element(tag)
        node.text = text
        node.tail = tail
        return node

    @staticmethod
    def addSibling(element, node):
        """set node as sibling of element
        """
        element.addnext(node)

    @staticmethod
    def insert_node(element, node):
        element.insert(0, node)

    @staticmethod
    def is_element_has_child(element):
        return len(element)!=0

    @staticmethod
    def is_element_text_none(element):
        return len(element.text) == 0

    @staticmethod
    def is_element_tail_none(element):
        return len(element.tail) == 0

    @staticmethod
    def is_element_content_none(element):
        return    len(element.text_content()) == 0

    @staticmethod
    def is_element_nest(element):
        """ <div>
                this is nest element
                <a> hello</a>
            </div>
        """
        if ElementHelper.is_element_has_child(element) and \
            len(ElementHelper.element_text_tail(element))!=len(ElementHelper.element_text_content(element)):
            return True
        return False

    @staticmethod
    def get_element_text_number(element):
        text = element.text
        return len(text)

    @staticmethod
    def get_element_content_number(element):
        text = ElementHelper.element_text_content(element)
        return len(text)

    @staticmethod
    def get_children(element, ignore_tag=['em','b','i','br','img']):
        """ get all children of element, include itself
        """
        if element.tag not in ignore_tag:
            children = [element]
        else:
            children = []
        for child in element:
            children.extend(ElementHelper.get_children(child))
        return children

    @staticmethod
    def get_element_depth(element):
        if len(element) == 0:
            return 1
        else:
            depths = [ElementHelper.get_element_depth(c) for c in element]
            return max(depths)+1

    @staticmethod
    def get_element_level(element):
        return int(element.get(px))

    @staticmethod
    def get_element_preorder_num(element):
        return int(element.get(py))

    @staticmethod
    def print_element(element_or_tree):
        from lxml import etree
        print (etree.tostring(element_or_tree, encoding='utf-8', pretty_print=True))


class TimeHelper(object):
    @staticmethod
    def url_timestamp():
        return str(time.strftime("%m-%d", time.localtime()))

class Random(object):

    @staticmethod
    def random_header():
        return random.choice(USER_AGENTS)





