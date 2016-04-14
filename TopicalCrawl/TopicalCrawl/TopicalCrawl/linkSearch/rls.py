#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: actlea
@file: rls.py
@time: 16-3-26 下午6:49
@description:
"""

import numpy as np


class rls:
    def __init__(self):
        self._lambda = 0.99
        self.c = 1E-4
        self.P = np.array([])
        self.w = np.array([])

    def evaluate(self, x):
        if self.w.size > 0:
            return np.dot(x, self.w)
        else:
            return np.zeros((x.shape[0], 1))

    def train(self, x, y):
        if self.w.size == 0:
            m = len(x)
            self.w = np.zeros((m, 1))
            self.P = np.identity(m) / self.c

        k = np.dot(self.P, x.T) / (self._lambda + np.dot(np.dot(x, self.P), x.T))
        z = y - np.dot(x, self.w)
        self.w = self.w + k * z
        self.P = (self.P - np.dot(np.dot(k, x), self.P)) / self._lambda


X = [1, 2, 3, 4]
X = np.array(X).reshape(-1, 1)
Y = [6, 5, 7, 10]
Y = np.array(Y).reshape(-1, 1)
N = X.shape[0]
r = rls()
Y_est = np.zeros((N, 1))
for i in range(0, N):
    Y_est[i] = r.evaluate(X[i, :])
    r.train(X[i, :], Y[i])

# print r.w
# print r.P
SE = (Y - Y_est) ** 2
print SE
print r.evaluate(6)
