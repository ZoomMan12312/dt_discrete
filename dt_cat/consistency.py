import numpy as np
import math

class ConsistencyInfo():
    def __init__(self, x_vectors, x_colNames, y_vector, tol=15):
        self.tol = tol
        self._x_vectors = x_vectors
        self._x_colNames = x_colNames
        self._y_vector = y_vector
        self._y_uniq = np.unique(y_vector)
        self._n = len(self._y_uniq)
        self._y_share = 1/self._n
        self._setVectorUniqueItems()
        self._getAllProbas()
        self._getAllConsistency()
        self._getAllFeatureConsistency()
        self._getTopFeature()

    def _setVectorUniqueItems(self):
        self.x_uniq = []
        for i in self._x_vectors:
            self.x_uniq.append(np.unique(i))

    def _getAllProbas(self):
        i = 0
        self.x_probas = []
        for vector in self._x_vectors:
            self.x_probas.append([])
            for attribute in self.x_uniq[i]:
                res = self._getProba(vector, attribute)
                self.x_probas[i].append(res)
            i += 1
    
    def _getProba(self, vector, attribute):
        probas = []
        probaSum = 0
        for y in self._y_uniq:
            match = 0
            cnt = 0
            for x in vector:
                if x == attribute and self._y_vector[cnt] == y:
                    match += 1
                cnt += 1
            proba = (match/cnt)*self._y_share
            probaSum += proba
            probas.append(proba)
        normalized = []
        for i in probas:
            normalized.append(i/probaSum)
        return normalized

    def _getAllConsistency(self):
        i = 0
        self.x_consistency = []
        for proba in self.x_probas:
            self.x_consistency.append([])
            for x in proba:
                res = self._getConsistency(x)
                self.x_consistency[i].append(res)
            i += 1

    def _getConsistency(self, probas):
        cs = []
        for proba in probas:
            if proba <= self._y_share:
                c = 1 - ( 1/( 1+math.exp( -10*self._n*proba + 5 ) ) )
            else:
                c = 1/( 1+math.exp( (5*self._n*(-2*proba + 1) +5) ) )
            cs.append(c)
        p = 1
        for c in cs:
            p *= c
        return p

    def _getAllFeatureConsistency(self):
        self.feature_consistency = []
        for cs in self.x_consistency:
            fc = self._getFeatureConsistency(cs)
            self.feature_consistency.append(fc)
    
    def _getFeatureConsistency(self, cs):
        m = len(cs)
        fc_mean = 0
        fc_max = 0
        fc_share = 1/m
        for c in cs:
            fc_mean += c/m
            if c > fc_max:
                fc_max = c
        fc_score = ((fc_max + fc_mean)/2)*( (2*(fc_share**(1/self.tol))) - (2*(0.5**(1/self.tol))) + 1)
        return fc_score

    def _getTopFeature(self):
        i = 0
        self.pairs = []
        self.top_feature = [0,'', 0]
        for f in self.feature_consistency:
            if f > self.top_feature[0]:
                self.top_feature = [f, self._x_colNames[i], i]
            self.pairs.append([f, self._x_colNames[i], i])
            i += 1
        self.sorted_pairs = sorted(self.pairs, key=lambda x: x[0], reverse=True)