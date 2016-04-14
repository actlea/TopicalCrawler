#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: link_util.py
@time: 16-3-22 上午8:59
@description:
"""
import random

from link_base import  *
from evaluator import *
from feature_builder import *


class ScoreDifferenceTotal(object):
    def __init__(self, k):
        self.k = k


    def score(self, node_evaluator, edge_evaluator, uf):
        pass



class EpsilonGreedyStrategy(object):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_next_node(self,frontier):
        """
        :param frontier: queue which save url uncrawlered
        :return: url
        """
        if random.random()>self.epsilon:
            return frontier.get()
        else:
            pass



class Crawler(object):
    def __init__(self, graph, evaluator, strategy, update_frequency, out_stream):
        self.evaluator = evaluator
        self.graph = graph
        self.strategy = strategy
        self.update_frequency = update_frequency
        self.out_stream = out_stream

    def write_data_to_stream(self, score):
        pass

    def crawl_node(self, nid, crawled_graph):
        weight = self.graph.get_node_weight(nid)
        #notice evaluator
        self.evaluator.add_to_trainings(crawled_graph, nid, weight)
        #get outgoing edges
        crawled_graph.add_vertex_unless_existing(nid, weight)
        for jt in self.graph.out_edges(nid):
            if crawled_graph.has_vertex(jt):
                if self.evaluator.contains_node(jt):
                    self.evaluator.mark_updated(jt)
                else:
                    pass
            else:
                self.evaluator.add_node(jt)
                self.evaluator.mark_updated(jt)
                crawled_graph.add_vertex_unless_existing(jt, 0, False)
            crawled_graph.add_edge(nid, jt, self.graph.get_edge_weight(nid, jt))

        return weight


    def crawl(self, seeds, steps=1000):
        crawled_graph = DirectedGraph()
        for nid in seeds:
            crawled_graph.add_vertex_unless_existing(nid, 0)

        score = 0.0
        for nid in seeds:
            score += self.crawl_node(nid, crawled_graph)
        self.evaluator.update_scores(crawled_graph)
        self.write_data_to_stream(score)

        #crawl
        while steps>=0 and self.evaluator.begin()!=self.evaluator.end():
            steps -= 1
            #call strategy to get next node
            next_node = self.strategy.get_next_node(self.evaluator.begin(), self.evaluator.end())
            score += self.crawl_node(next_node, crawled_graph)
            #remove node from the frontier
            self.evaluator.frontier.remove_node(next_node)
            #update evaluator
            if steps % self.update_frequency==0:
                self.evaluator.update_scores(crawled_graph)
        self.write_data_to_stream(score)
        self.out_stream.write('\n')


class CrawlerWithFs(Crawler):
    def __init__(self, graph, evaluator, strategy, update_frequency, out_stream, score):
        Crawler.__init__(self,graph, evaluator, strategy, update_frequency, out_stream)
        self.score = score

    def write_data_to_stream(self, score):
        pass







def load_seed(seed_file):
    with open(seed_file) as fr:
        return fr.readlines()



def load_feature_builder(feature_builders):
    feature_builders.append(AverageFeatureBuilder())



def load_evaluators(evaluators, graph):
    """
    :param evaluators: a list of evaluator
    :param graph: directed graph
    :return:
    """
    # 0 -oracle
    # evaluators.append(OracleEvaluator(graph))
    #1 - sum

    #NR

    #LinRegFL
    fb = FirstOrderFeatureBuilder()


    #LinReg SL

    #OPIC

    #mab

    #SVM
    # avg = AverageFeatureBuilder()
    # evaluators.append(SVMEvaluator(avg))



def precisin(graph_file, seed_file, update_frequency=4, steps = 1000, delimiter='\t'):
    print '---------load graph file------------'
    directed_graph = DirectedGraph()
    directed_graph.load_graph_from_file(graph_file)
    print '--------------choose score function----------'
    score = None

    print '-----------pick greedy strategy-------------------'
    strat = EpsilonGreedyStrategy(0)
    print '-----------load feature builder-------------------'
    feature_builders = []
    load_feature_builder(feature_builders)
    print '-----------load evaluators-------------------'
    evaluators = []
    load_evaluators(evaluators, directed_graph)

    print '-----------load seeds and crawl-------------------'
    seedlist = load_seed(seed_file)
    for line in seedlist:
        seeds = line.split(delimiter)
        print '------------------start crawl------------'
        c = CrawlerWithFs(directed_graph, evaluators, strat, update_frequency, score)
        c.crawl(seeds, steps)


def test_feature_builder():
    graph_file = './test/graph'
    seed_file = './test/seeds'
    directed_graph = DirectedGraph()
    directed_graph.load_graph_from_file(graph_file)

    featur_builder = FirstOrderSecondLevelFeatureBuilder(False)
    for node in directed_graph.get_nodes():
        nid = node
        score = directed_graph.get_node_weight(nid)
        fv_sl = featur_builder.construct_features(directed_graph, nid)
        print 'nid:%s, score: %d' %(nid, score)
        for feature in fv_sl:
            print 'feature:%d \t' %feature


def test_crawler():
    graph_file = './test/graph'
    seed_file = './test/seeds'
    g = DirectedGraph()
    g.load_graph_from_file(graph_file)
    strat = EpsilonGreedyStrategy(0)
    evaluators = [OracleEvaluator(g), BFSEvaluator()]

    score = ScoreDifferenceTotal(1)

    _eval = PickFirstEvaluator(evaluators)

    seed_list = load_seed(seed_file)
    out_stream = open('tmp.txt', 'w')
    for line in seed_list:
        line  =line.strip()
        seeds = line.split()
        c = CrawlerWithFs(g, _eval, strat, 1, out_stream, score)
        c.crawl(seeds, 10)
    out_stream.close()







if __name__=='__main__':
    graph_file = './test/graph'
    seed_file = './test/seeds'
    # precisin(graph_file, seed_file, update_frequency=4, steps = 1000, delimiter='\t')
    # test_feature_builder()
    test_crawler()