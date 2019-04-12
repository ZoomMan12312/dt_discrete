#layered decision tree
import pandas as pd
import numpy as np
from dt_cat.consistency import ConsistencyInfo

class Tree():
    def __init__(self, x_vectors, x_colNames, y_vector, tol=15, max_layers=100):
        self.root_node = _Node(x_vectors, x_colNames, y_vector, tol, 0, max_layers, None, None)
        nodes = 0

    def predict(self, x_vector, x_colNames):
        return self.root_node.predict(x_vector, x_colNames)

class _Node():
    def __init__(self, x_vectors, x_colNames, y_vector, tol, layer, max_layers, by_attribute, by_feature):
        #IS NODE PURE??!?!?!?!
        #do next instructions only if not pure and len(x_vectors) > 1
        self._y_vector = y_vector
        self.y_uniq = np.unique(y_vector)
        self._attribute = by_attribute
        self._max_layers = max_layers
        self._tol = tol
        self.layer = layer
        self._feature = by_feature

        self._getProbas()

        if len(self.y_uniq) == 1:
            self.end_node = True
            self.pure = True
            self.cat = self.y_uniq

        else:
            self.pure = False
            if len(x_vectors) >= 1 and self.layer <= max_layers:
                self.end_node = False
                self._buildNodes(x_vectors, x_colNames, y_vector)
            else:
                self.end_node = True

        # print(layer)
        # print(self.pure)
        # print(self._attribute)
        # print(self._feature)
        # if self.pure:
        #     print(self.cat)
        # print('--------------------------------')

    def _buildNodes(self, x_vectors, x_colNames, y_vector):
        #1.find best node to split on
        ci = ConsistencyInfo(x_vectors, x_colNames, y_vector, self._tol)
        self.top_feature = ci.top_feature
        self.top_feature_name = ci.top_feature[1]
        self.child_nodes = []
        #2.Get unique x values
        x_uniq = ci.x_uniq[self.top_feature[2]]

        #3.test
        for i in x_uniq:
            self.child_nodes.append(self._buildSingleNode(x_vectors, x_colNames, y_vector, self.top_feature[2], i))

    def _buildSingleNode(self, x_vectors, x_colNames, y_vector, top_feature_index, attribute):
        x_transposed = x_vectors.transpose()
        #y_transposed = y_vector.transpose()
        x_df = pd.DataFrame(x_transposed)
        #y_df = pd.DataFrame(y_transposed)
        y_df = pd.DataFrame({'y':y_vector})
        df = pd.concat([x_df, y_df], axis=1)
        df = df[df[top_feature_index] == attribute]
        y = df['y'].values
        x = df.drop(['y'], axis=1).values.transpose()
        x = np.delete(x, [top_feature_index], axis=0)
        x_col = np.delete(x_colNames, top_feature_index)
        return _Node(x, x_col, y, self._tol, self.layer + 1, self._max_layers, attribute, x_colNames[top_feature_index])
        #print(x)
        #print(df[top_feature_index])
        
    def _getProbas(self):
        cnt = len(self._y_vector)
        self._probas = []
        for y in self.y_uniq:
            match = 0
            for i in self._y_vector:
                if y == i:
                    match += 1
            self._probas.append([y, match/cnt])

    def predict(self, x_vector, x_colNames):
        if self.end_node == True:
            return self._probas
        else:
            i = np.where(x_colNames == self.top_feature_name)
            for node in self.child_nodes:
                if node._attribute == x_vector[i]:
                    return node.predict(x_vector, x_colNames)
            return self._probas