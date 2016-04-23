#!/usr/bin/env python
# encoding: utf-8

"""
@version: ??
@author: phpergao
@license: Apache Licence 
@file: link_relevance.py
@time: 16-4-23 下午9:14
"""

from collections import Counter

from classifier.pure_url_classify import url_predict
from classifier.text_classifier import anchor_predict

import time


def compute_link_relevance(url_item, url_classifier, anchor_classifier):
    anchor_text = url_item['anchor_text']
    url = url_item['url']

    pred_u, prob_u = url_predict(url, url_classifier)
    pred_a, prob_a = anchor_predict(anchor_text, anchor_classifier)

    atom = [(pred_u, prob_u), (pred_a, prob_a)]
    result = [p for p in atom if p[1] >= 0.5]

    labels = [p[1] for p in result]

    label_probs = {}
    for r in result:
        label_probs.setdefault(r[0], []).append(r[1])

    counter = Counter(labels).most_common()

    res_label = counter[0][0]
    res_prob = max(label_probs[res_label])

    return res_label, res_prob


def put_to_graph(url_item, graph, url_relevance_dict):
    '''

    :param url_item: base_url, node_weight, url, anchor_text, prob
    :param visited_node: {base_url:weight}
    :param graph: {url:[parent node]}
    :param url_relevance_dict:
    :return:
    '''
    url = url_item['url']
    base_url = url_item['base_url']
    node_weight = url_item['node_weight']
    prob = url_item['prob']
    graph.setdefault(url, []).append((base_url, node_weight))
    if prob > url_relevance_dict.get(url, 0.0):
        url_relevance_dict[url] = prob
    else:
        url_item['prob'] =  url_relevance_dict.get(url, 0.0)



def compute_link_priority1(graph, url_item, belta=0.25):
    url = url_item['url']
    parents = graph[url]
    weight = 1.0*sum([n[1] for n in parents])/len(parents)

    p = belta * weight + (1.0-belta)*url_item['prob']
    return p


def compute_link_priority2(url_item,belta=0.25):
    priority = belta * url_item['node_weight'] + (1.0 - belta) * url_item['prob']
    return priority


def url_control(url_item,
                url_classifier,
                anchor_classifier,
                level0,
                level1,
                graph,
                url_relevance_dict,
                threshold=0.8,
                belta=0.25):
    url = url_item['url']
    if url in level0:
        return None

    label, prob = compute_link_relevance(url_item, url_classifier, anchor_classifier)
    url_item['prob'] = prob
    if label=='0' and prob>=threshold:
        #put into the first level
        if url in level1:
            put_to_graph(url_item, graph, url_relevance_dict)
            priority = compute_link_priority1(graph, url_item, belta)
            del level1[url]
            del graph[url]
            del url_relevance_dict[url]
        else:
            priority = belta*url_item['node_weight']+(1.0-belta)*prob
        url_item['priority'] = priority
        level0[url]=url_item

    elif label=='0' and threshold>prob>=0.5:
        url_item['prob'] = prob
        #put into the second level
        #1.compute url value
        put_to_graph(url_item, graph, url_relevance_dict)
        if url in level1:
            url_item['priority'] = compute_link_priority1(graph, url_item, belta)
        else:
            url_item['priority'] = compute_link_priority2(url_item, belta)
        level1[url] = url_item
    else:
        return


def get_next_url(level0, level1, graph, url_relevance):
    if len(level0)>0:
        tmp = sorted(level0.items(), key=lambda x:x['priority'], reverse=True)
        del level0[tmp[0]]
        return tmp[0], tmp[1]['priority']

    if len(level1)>0:
        tmp = sorted(level1.items(), key=lambda x: x['priority'], reverse=True)
        del level0[tmp[0]]
        del graph[tmp[0]]
        del url_relevance[tmp[0]]
        return tmp[0], tmp[1]['priority']


    time.sleep(50)






