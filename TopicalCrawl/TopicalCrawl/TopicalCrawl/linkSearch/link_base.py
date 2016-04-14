#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: link_base.py
@time: 16-3-22 上午10:43
@description:
"""

class Gnode(object):
    def __init__(self, nid, weight=0):
        self.nid = nid
        self.weight = weight


class Gedge(object):
    def __init__(self, fid, tid, weight):
        self.fid = fid
        self.tid = tid
        self.weight = weight


class DirectedGraph(object):
    def __init__(self):
        self.node__dict = {} #{nid:weight}
        self.edge__dict = {} #{(from, to):weight}
        self.in_edges = {} #{nid:[]} 指向nid的节点
        self.out_edges = {} #{nid:[]}nid指向的节点


    def add_vertex_unless_existing(self, node, weight,  update_weight=True):
        if self.has_vertex(node):
            if update_weight:
                self.node__dict[node] = weight
        else:
            self.node__dict[node] = weight


    def add_edge(self, fromvd, tovd, weight):
        self.edge__dict[(fromvd, tovd)] = weight
        self.out_edges.setdefault(fromvd, set()).add(tovd)
        self.in_edges.setdefault(tovd, set()).add(fromvd)


    def get_node_weight(self, nid):
        return self.node__dict.get(nid, 0)


    def get_edge_weight(self, fid, tid):
        return self.edge__dict.get((fid, tid), 0)

    def get_in_edges(self, nid):
        return self.in_edges[nid]

    def get_out_edges(self, nid):
        return self.out_edges[nid]

    def get_nodes(self):
        return self.node__dict.keys()


    def has_vertex(self, node):
        if node in self.node__dict:
            return True
        return False


    def load_graph_from_file(self, graph_file, delimiter='\t'):
        cnt = 0
        with open(graph_file, 'r') as fr:
            for line in fr.readlines():
                cnt += 1
                if cnt == 1001: break
                line = line.strip()
                line = line.split(delimiter)
                if len(line)==2:
                    fromid, weight = line
                    weight = float(weight)
                    self.add_vertex_unless_existing(fromid, weight)
                elif len(line)==3:
                    fromid, toid, weight = line
                    weight = float(weight)
                    self.add_vertex_unless_existing(fromid, 0)
                    self.add_vertex_unless_existing(toid, 0)
                    self.add_edge(fromid, toid, weight)
                else:
                    continue


def pre_process(train_file1,train_file2, out_file, delimiter='\t'):
    node_weight = {}







# class FrontierScore(object):
#     def __init__(self):pass
#     def score(self,eval1, eval2):pass
#     def score(self,begin1,end1, begin2, end2):pass
#
#
#
# class ScoreDifferenceTotal(FrontierScore):
#     def __init(self, k):
#         super(FrontierScore, self).__init__()
#         self.k = k
#
#     def score(self, eva1, eva2):
#         begin1 = eva1.begin()
#         end1 = eva1.end()
#         begin1 = eva2.begin()
#         end2 = eva2.end()
#
#         #get sum of oracle scores for the oracle top k, sum1
#         sum1 = 0
#         _k = self.k
#         for it in eva1[begin1:end1]:
#             if _k==0:break
#             sum1 += 1


if __name__ =='__main__':
    # src_file= '/mnt/UbutunShare/graduate/DataSet/scrapy_dataset/qq/url.sample.txt'
    # out_file = 'graph.txt'
    # pre_process(src_file, out_file)
    pass

