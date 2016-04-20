#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
@author: actlea
@file: evaluator.py
@time: 16-3-22 下午3:02
@description:
"""
import time
import math



class Weight_Nodes(object):
    def __init__(self, nid, score=-1):
        self.nid = nid
        self.score = score


class Frontier(object):
    def __init__(self):
        self.weighted_nodes = {}

    def set_node(self, nid, score=-1):
        self.weighted_nodes[nid] = score

    def remove_node(self, nid):
        if nid in self.weighted_nodes:
            del self.weighted_nodes[nid]

    def get_node_score(self, nid):
        return self.weighted_nodes.get(nid, -1)

    def clear(self):
        self.weighted_nodes.clear()

    def contains(self, nid):
        return nid in self.weighted_nodes

    def get(self):
        sorted_node = sorted(self.weighted_nodes.items(), key=lambda  x:x[1], reverse=True)
        return sorted_node[0][0]

    def begin(self):pass
    def end(self):pass



class Evaluator(object):
    def __init__(self):
        self.generator = None  #random generator
        self.updated = set()

    def mark_updated(self, nid):
        self.updated.add(nid)

    def update_scores(self, graph):
        self.updated.clear()

    def clear_all(self):
        self.updated.clear()

    def add_to_trainings(self, graph, nid, score):pass
    def add_node(self, nid):pass
    def begin(self):pass
    def end(self):pass



class FibonacciHeapEvaluator(Evaluator):
    def __init__(self, frontier):
        super(Evaluator, self).__init__()
        self.frontier = frontier

    def clear_all(self):
        self.frontier.clear()
        Evaluator.clear_all()

    def begin(self):
        return self.frontier.begin()

    def end(self):
        return self.frontier.end()

    def remove_node(self, nid):
        self.frontier.remove_node(nid)

    def add_node(self, nid):
        self.frontier.set_node(nid, -1)

    def contains(self, nid):
        return self.frontier.contains(nid)

    def get_node_score(self,nid):
        return self.frontier.get_node_score(nid)



class BFSEvaluator(FibonacciHeapEvaluator):
    def __init__(self):
        super(FibonacciHeapEvaluator, self).__init__()

    def update_scores(self, graph):
        for node in self.updated:
            self.frontier.set_node(node, 1.0)

        self.update_scores(graph)




class OracleEvaluator(FibonacciHeapEvaluator):
    def __init__(self, original_graph):
        super(FibonacciHeapEvaluator, self).__init__()
        self.original_graph = original_graph

    def update_scores(self, graph):
        for node in self.updated:
            prediction = self.original_graph.get_node_weight(node)
            self.frontier.set_node(node, prediction)



class PickFirstEvaluator(Evaluator):
    def __init__(self, evaluators):
        super(Evaluator, self).__init__()
        self.evaluators = evaluators
        self.running_times = dict()

    def mark_updated(self, nid):
        for eval in self.evaluators:
            eval.mark_updated(nid)
        Evaluator.mark_updated(nid)


    def clear_all(self):
        for eval in self.evaluators:
            eval.clear_all()


    def add_to_trainings(self, graph, nid, score):
        i = 0
        for eval in self.evaluators:
            first_time = time.time()
            eval.add_to_training(graph, nid, score)
            second_time = time.time()

            self.running_times[i] = second_time-first_time
            i += 1

    def update_scores(self, graph):
        i = 0
        for _eval in self.evaluators:
            first_time = time.time()
            _eval.update_scores(g)
            second_time = time.time()

            self.running_times[i] = second_time-first_time
            i += 1

    def remove_node(self, nid):
        for eval in self.evaluators:
            eval.remove_node(nid)

    def add_node(self, nid):
        for eval in self.evaluators:
            eval.add_node(nid)


    def get_node_score(self, nid):
        return self.evaluators[0].get_node_score(nid)

    def get_evaluators(self):
        return self.evaluators


class LinearRegressionEvaluator(FibonacciHeapEvaluator):
    def __init__(self, feature_builder, use_logs=False):
        super(FibonacciHeapEvaluator, self).__init__()
        self.feature_builder = feature_builder
        self.logs = use_logs

        self.training_data = {}

    def add_to_trainning(self, graph, nid, score):
        FibonacciHeapEvaluator.add_to_trainings(graph, nid, score)
        if score!=0:
            drop_vector = False
            fvector = self.feature_builder.construct_features(graph, nid)
            if self.logs:
                idx = 0
                for feature in fvector:
                    if feature>0:
                        fvector[idx] = math.log(feature)
                    else:
                        drop_vector = True
                    idx += 1
            if not drop_vector:
                if self.logs:
                    score = math.log(score)
                self.training_data[nid] = {fvector:score}


    def update_scores(self, graph):
        self.training_data.clear()
        FibonacciHeapEvaluator.update_scores(graph)

def log(score):
    if score <= 0:
        return score
    else:
        return math.log(score)


class DLibLinearRegressionEvaluator(LinearRegressionEvaluator):
    def __init__(self, feaure_builder,model, ff=1, use_logs=False):
        LinearRegressionEvaluator.__init__(self, feaure_builder, use_logs)
        self.forget_factor = ff
        self.has_model = False
        self.model = model

    def feature_vector_to_matrix(self, feature_vector):
        #把feature转换为模型参数的形式
        return self.feature_builder

    def update_scores(self, graph):
        #利用模型对特征的值做出预测
        for p in self.training_data:
            #{nid:{fvector:score}}
            features, score = self.training_data[p].items()
            if self.logs:
                if score>0:
                    score = math.log(score)
                else:
                    score = 0
            self.model.train(self.feature_vector_to_matrix(features), score)
            self.has_model = True
        for nid in self.updated:
            fvector = self.feature_builder.construct_features(graph, nid)
            if(self.logs):
                idx = 0
                for feature in fvector:
                    fvector[idx] = log(feature)
                    idx += 1
            if self.has_model:
                prediction = self.model.predict(self.feature_vector_to_matrix(fvector))
            else:
                prediction = 0
            if self.logs:
                prediction = math.exp(prediction)
            self.frontier.set_node(nid, prediction)

    def clear_all(self):
        self.has_model = False
        LinearRegressionEvaluator.clear_all()




class SVMEvaluator(FibonacciHeapEvaluator):
    def __init__(self, feature_builder):
        self.feature_builder = feature_builder
        self.training_data = {} #{nid:[]}

    def add_to_training(self, graph, nid, score):
        FibonacciHeapEvaluator.add_to_trainings(graph, nid, score)
        fvector = self.feature_builder.construct_features(graph, nid)
        self.training_data[nid] = {fvector:score}


    def update_scores(self, graph):
        pass


    def clear_all(self):
        pass






















