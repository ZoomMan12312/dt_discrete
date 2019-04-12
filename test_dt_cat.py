import pandas as pd
import numpy as np
from dt_cat.dt import Tree

def run():
    data = pd.read_csv('mushrooms.csv')
    y = data['class']
    y_name = y.name
    y_vector = y.values
    
    x = data.drop(['class'], axis=1)
    x_colNames = x.columns.values
    x_values = x.values
    x_vectors = x_values.transpose()

    tree = Tree(x_vectors, x_colNames, y_vector)

    # res = tree.predict(np.array(['a']), np.array(['odor']))
    res = tree.predict(np.array(['0']), np.array(['0']))
    print(res)


run()