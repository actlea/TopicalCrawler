#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: feature_builder.py
@time: 16-3-22 上午10:29
@description:
"""

import math



class FeaturBuilder(object):
    def __init__(self, logs=False):
        self.logs = logs

    def construct_features(self,graph, node):
        pass


    def get_number_features(self):
        pass


class AverageFeatureBuilder(FeaturBuilder):
    def __init__(self):
        super(FeaturBuilder, self).__init__()

    def get_number_features(self):
        return 3

    def construct_features(self,graph, nid):
        """
        :param graph: 遍历第一层节点，即指向目标的节点
        :param node:
        :return:
        """
        count = 0
        features = [0 for i in range(self.get_number_features())]
        for jt in graph.get_in_edges(nid):
            node_weight = graph.get_node_weight(jt)
            edge_weight = graph.get_edge_weight(jt, nid)

            features[0] += node_weight
            features[1] += edge_weight
            features[2] += 1.0*node_weight*edge_weight
            count += 1

        if count!=0:
            for i in range(3):
                features[i] = 1.0*features[i]/count

        return features


class FirstOrderFeatureBuilder(FeaturBuilder):
    def __init__(self, logs=True):
        FeaturBuilder.__init__(self, logs)



    def get_number_features(self):
        return 3


    def construct_features(self,graph, nid):
        """
        :param graph: 遍历第一层节点，即指向目标的节点
        :param node:
        :return:
        """
        features = [0 for i in range(self.get_number_features())]
        for jt in graph.get_in_edges(nid):
            node_weight = graph.get_node_weight(jt)
            edge_weight = graph.get_edge_weight(jt, nid)

            features[0] += node_weight
            features[1] += edge_weight
            features[2] += 1.0*node_weight*edge_weight
        if self.logs:
            for i in range(self.get_number_features()):
                features[i] = math.log(1.0+features[i])

        return features



class FirstOrderSecondLevelFeatureBuilder(FeaturBuilder):
    def __init__(self, logs=False):
        # super(FeaturBuilder(logs), self).__init__()
        FeaturBuilder.__init__(self, logs)



    def get_number_features(self):
        return 7

    def construct_features(self,graph, nid):
        """
        :param graph: 遍历第一层节点，即指向目标的节点
        :param node:
        :return:
        """
        features = [0 for i in range(self.get_number_features())]
        for jt in graph.get_in_edges(nid):
            node_weight = graph.get_node_weight(jt)
            edge_weight = graph.get_edge_weight(jt, nid)

            features[0] += 1
            features[1] += node_weight
            features[2] += edge_weight
            features[3] += 1.0*node_weight*edge_weight
            for ot in graph.get_out_edges(jt):
                node_weight = graph.get_node_weight(ot)
                edge_weight = graph.get_edge_weight(jt, ot)

                features[4] += node_weight
                features[5] += edge_weight
                features[6] += node_weight*edge_weight

        return features




class IEINFeatureBuilder(FeaturBuilder):
    def __init__(self):
        super(FeaturBuilder, self).__init__()

    def get_number_features(self):
        return 2

    def construct_features(self,graph, nid):
        """
        :param graph: crawled graph
        :param node:
        :return:
        """
        features = [0 for i in range(self.get_number_features())]
        for jt in graph.get_in_edges(nid):
            node_weight = graph.get_node_weight(jt)
            edge_weight = graph.get_edge_weight(jt, nid)

            features[0] += node_weight
            features[1] += edge_weight

        return features



class MaxFeatureBuilder(FeaturBuilder):
    def __init__(self):
        super(FeaturBuilder, self).__init__()

    def get_number_features(self):
        return 3

    def construct_features(self,graph, nid):
        """
        :param graph: 遍历第一层节点，即指向目标的节点
        :param node:
        :return:
        """
        features = [0 for i in range(self.get_number_features())]
        for jt in graph.get_in_edges(nid):
            node_weight = graph.get_node_weight(jt)
            edge_weight = graph.get_edge_weight(jt, nid)

            if node_weight>features[0]:
                features[0] = node_weight
            if edge_weight>features[1]:
                features[1] = edge_weight
            if node_weight*edge_weight>features[2]:
                features[2] = 1.0*node_weight*edge_weight

        return features














