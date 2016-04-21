#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: actlea
@file: cetd.py
@time: 16-3-3 下午3:43
"""
from math import log as qLn
from math import exp as qExp


from util import ElementHelper, StringHelper
from config import *


def search_tag(element, attribute, value):
    d2s_value = str(value)
    target = element
    attr_value = float(element.get(attribute))
    if (attr_value-value>-1*float_epsilon) and \
            (attr_value-value<float_epsilon):
        return target

    #search the max_densit_sum element using xpath selector
    xpath_selector = './/*[@' + attribute + '="' + d2s_value +'"]'
    target = element.find(xpath_selector)
    return target

def CountChar(element):
    char_num = len(ElementHelper.element_text_content(element))
    l2s_char_num = str(char_num)
    element.set(kg_char_num, l2s_char_num)
    for child in element:
        CountChar(child)

def CountTag(element):
    tag_num = 0
    l2s_tag_num = str(tag_num)
    if ElementHelper.is_element_has_child(element):
        for child in element:
            CountTag(child)
        for child in element:
            tag_num += long(child.attrib.get(kg_tag_num))+1
        l2s_tag_num = str(tag_num)
        element.set(kg_tag_num, l2s_tag_num)
    else:
        element.set(kg_tag_num, l2s_tag_num)

def UpdateLinkChar(element):
    for child in element:
        child.set(kg_linkchar_num, child.get(kg_char_num))
        UpdateLinkChar(child)

def CountLinkChar(element):
    '''call this function after CountChar'''
    linkchar_num = 0
    tag_name = element.tag
    for child in element:
        CountLinkChar(child)
        #deal with hyperlink and sth like that
    if tag_name=='a' or tag_name == 'button' or tag_name =='select':
        linkchar_num = long(element.attrib.get(kg_char_num,0))
        UpdateLinkChar(element)
    else:
        for child in element:
            linkchar_num += long(child.attrib.get(kg_linkchar_num, 0))
    l2s_linkchar_num  = str(linkchar_num)
    element.set(kg_linkchar_num, l2s_linkchar_num)

def UpdateLinkTag(element):
    for child in element:
        child.set(kg_linktag_num, child.attrib.get(kg_tag_num))
        UpdateLinkTag(child)

def CountLinkTag( element):
    '''call this function after CountChar, CountLinkChar'''
    linktag_num = 0
    tag_name = element.tag

    for child in element:
        CountLinkTag(child)

    # deal with hyperlink or sth like that
    if tag_name == 'a' or tag_name=='button' or tag_name=='select':
        linktag_num =long(element.attrib.get(kg_tag_num))
        UpdateLinkTag(element)
    else:
        for child in element:
            linktag_num += long(child.attrib.get(kg_linktag_num))
            tag_name = child.tag
            #if a tag is <a> or sth plays similar role in web pages, then anchor number add 1
            if tag_name == 'a' or tag_name=='button' or tag_name=='select':
                linktag_num+=1
            else:
                child_linktag_num = long(child.attrib.get(kg_linktag_num))
                child_tag_num = long(child.attrib.get(kg_tag_num))
                child_char_num = long(child.attrib.get(kg_char_num))
                child_linkchar_num = long(child.attrib.get(kg_linkchar_num))
                #child_linktag_num != 0: there are some anchor under this child
                if child_linktag_num == child_tag_num and \
                                child_char_num == child_linkchar_num and \
                                0 != child_linktag_num:
                    linktag_num+=1

    l2s_linktag_num = str(linktag_num)
    element.set(kg_linktag_num, l2s_linktag_num)

def ComputeTextDensity(element, ratio):
    '''call this function after CountChar, CountTag, CountLinkChar, CountLinkTag'''
    char_num = long(element.attrib.get(kg_char_num))
    tag_num = long(element.attrib.get(kg_tag_num))
    linkchar_num = long(element.attrib.get(kg_linkchar_num))
    linktag_num = long(element.attrib.get(kg_linktag_num))

    text_density = 0.0
    d2s_text_density = str(0.0)

    if 0 == char_num:
        text_density = 0
    else:
        un_linkchar_num = char_num - linkchar_num
        if 0 == tag_num:
            tag_num = 1
        if 0 == linkchar_num:
            linkchar_num = 1
        if 0 == linktag_num:
            linktag_num = 1
        if 0 == un_linkchar_num:
            un_linkchar_num = 1

        a = (1.0 * char_num / tag_num)
        b = (1.0 * char_num * tag_num) / (1.0 * linkchar_num * linktag_num)
        c = 1.0 * char_num * linkchar_num / un_linkchar_num + ratio * char_num + qExp(1.0)
        try:
            text_density = a * qLn(b) / qLn(qLn(c))

        except(ValueError):
            print 'Errors, tag:%s, char_num:%s, tag_num:%s, linkchar_num:%s, linktag_num:%s,' \
                  'a:%s, b:%s, c:%s' %(element.tag, char_num, tag_num, linkchar_num, linktag_num, str(a),str(b),str(c))

    # convert double to str
    d2s_text_density = str(text_density)
    element.set(kg_text_density, d2s_text_density)

    for child in element:
        ComputeTextDensity(child, ratio)

def ComputeDensitySum(element, ratio):
    density_sum, char_num_sum  = 0.0, 0
    _from, index, length = 0, 0, 0

    content = ElementHelper.element_text_content(element)
    if ElementHelper.is_element_has_child(element):
        for child in element:
            ComputeDensitySum(child, ratio)
        for child in element:
            density_sum += float(child.attrib.get(kg_text_density))
            char_num_sum += long(child.attrib.get(kg_char_num))

            #text before tag
            child_content = ElementHelper.element_text_content(child)
            index = -1
            if child_content != '':
                index = StringHelper.index_of(content, child_content, _from)

            if index > -1:
                length = index - _from
                if length > 0:
                    try:
                        tmp = length * qLn(1.0 * length) / qLn(qLn(ratio * length + qExp(1.0))) #此处的计算结果都为0
                        density_sum += tmp
                    except ZeroDivisionError:
                        pass
                _from = index + len(child_content)

        #text after tag
        length = len(ElementHelper.element_text_content(element)) - _from
        if length>0:
            try:
                density_sum += length * qLn(1.0 * length) / qLn(qLn(ratio * length + qExp(1.0)))
            except ZeroDivisionError:
                pass
    else:
        density_sum = float(element.attrib.get(kg_text_density))

    d2s_density_sum  = str(density_sum)
    element.set(kg_density_sum, d2s_density_sum)

def FindMaxDensitySum(element):
    max_density_sum = float(element.get(kg_density_sum))

    for child in element:
        temp_max_density_sum = FindMaxDensitySum(child)
        import sys
        if (temp_max_density_sum - max_density_sum > sys.float_info.epsilon):
            max_density_sum = temp_max_density_sum

    #record the max_density_sum under the element
    d2s_max_density_sum = str(max_density_sum)
    element.set(kg_max_density_sum, d2s_max_density_sum)
    return max_density_sum

def SetMark(element, mark):
    i2s_mark = str(mark)
    element.set(kg_mark, i2s_mark)
    for child in element:
        SetMark(child, mark)

def GetThreshold(element, max_density_sum):
    threshold = -1.0
    #search the max density sum element
    #1.先找到density_sum最大的节点
    target = search_tag(element, kg_density_sum, max_density_sum)

    #2.设置初始text-density的阈值
    threshold = float(target.get(kg_text_density))
    SetMark(target, 1)

    #3.然后查找density_sum最大的节点到BODY节点之间最小的text-density作为阈值
    parent = target.getparent()
    while True:
        if parent.tag != 'html':
            text_density = float(parent.get(kg_text_density))
            if((threshold - text_density) > -1 * float_epsilon):
                threshold = text_density
            parent.set(kg_mark,'2')
            parent = parent.getparent()
        else:
            break
    return threshold

def FindMaxDensitySumTag(element, max_density_sum):
    # search the max densitysum element
    target = search_tag(element, kg_density_sum, max_density_sum)
    # print target.tag,":",target.get(kg_density_sum)
    mark = long(target.get(kg_mark))
    if 1 == mark:
        return
    SetMark(target, 1)
    parent = target.getparent()
    while True:
        if parent.tag != 'html':
            parent.set(kg_mark, "2")
            parent = parent.getparent()
        else:
            break

def MarkContent(element, threshold):
    text_density = float(element.get(kg_text_density))
    max_density_sum = float(element.get(kg_max_density_sum))
    mark = long(element.get(kg_mark))

    if mark != 1 and (text_density - threshold > -1 * float_epsilon):
        FindMaxDensitySumTag(element, max_density_sum)
        for child in element:
            MarkContent(child, threshold)

def RemoveAttribute(element):
    del element.attrib[kg_char_num]
    del element.attrib[kg_tag_num]
    del element.attrib[kg_linkchar_num]
    del element.attrib[kg_linktag_num]
    del element.attrib[kg_text_density]
    del element.attrib[kg_density_sum]
    del element.attrib[kg_max_density_sum]
    del element.attrib[kg_mark]
    for child in element:
        RemoveAttribute(child)



def cetd_parse(doctree):
    body = ElementHelper.get_body(doctree)
    CountChar(body)
    CountTag(body)
    CountLinkChar(body)
    CountLinkTag(body)
    char_num = float(body.attrib.get(kg_char_num))
    if char_num==0: char_num=1
    linkchar_num = float(body.attrib.get(kg_linkchar_num))
    ratio = linkchar_num/char_num
    ComputeTextDensity(body, ratio)
    ComputeDensitySum(body, ratio)
    max_density_sum = FindMaxDensitySum(body)
    SetMark(body, 0)
    thresold = GetThreshold(body, max_density_sum)
    MarkContent(body, thresold)

def get_page_link_ratio(doctree):
    body = ElementHelper.get_body(doctree)
    CountChar(body)
    CountTag(body)
    CountLinkChar(body)
    CountLinkTag(body)
    char_num = float(body.attrib.get(kg_char_num))
    if char_num==0: char_num=1
    linkchar_num = float(body.attrib.get(kg_linkchar_num))
    ratio = linkchar_num/char_num

    return ratio


def CleanTreeByMark(element):
    mark = long(element.get(kg_mark))
    if 0==mark:
        ElementHelper.remove_element(element)
    elif 1==mark:
        return
    else:
        for child in element:
            CleanTreeByMark(child)


def get_aricle_cetd(doctree):
    cetd_parse(doctree)
    body = ElementHelper.get_body(doctree)
    # ElementHelper.print_element(body)
    CleanTreeByMark(body)
    RemoveAttribute(body)
    return ElementHelper.element_text_content(body)


if __name__=='__main__':
    # dir = '/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/other_neg/original/42.html'
    # dir = '/mnt/UbutunShare/graduate/DataSet/original/0.htm'
    dir = '/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/ifeng/image/24.html'
    dir = '/mnt/UbutunShare/Work/CETD_DATA/wiki/original/23.htm'
    from api import HtmlHelper
    from util import FileHelper
    content = FileHelper.readUTF8File(dir)
    doc = HtmlHelper.create_doc(content, 'utf-8')
    print get_aricle_cetd(doc)