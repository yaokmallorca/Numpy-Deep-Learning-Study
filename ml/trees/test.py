import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from trees import DecisionTree

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

def create_data():
    datasets = [['y', 'no', 'no', 'nb', 0],
                ['y', 'no', 'no', 'good', 0],
                ['y', 'yes', 'no', 'good', 1],
                ['y', 'yes', 'yes', 'nb', 1],
                ['y', 'no', 'no', 'nb', 0],
                ['m', 'no', 'no', 'nb', 0],
                ['m', 'no', 'no', 'good', 0],
                ['m', 'yes', 'yes', 'good', 1],
                ['m', 'no', 'yes', 'very good', 1],
                ['m', 'no', 'yes', 'very good', 1],
                ['o', 'no', 'yes', 'very good', 1],
                ['o', 'no', 'yes', 'good', 1],
                ['o', 'yes', 'no', 'good', 1],
                ['o', 'yes', 'no', 'very good', 0],
                ['o', 'no', 'no', 'nb', 0],
               ]
    catgory = [u'age', u'work', u'own house', u'credit', u'catgory']
    return datasets, catgory

datasets, category = create_data()
datasets = np.array(datasets)
train_data = datasets[:, :-1]
labels = datasets[:, -1]

# trees 
dt = DecisionTree(0.01)
decision_tree = dt.fit(train_data, labels, category)
print(dt.predict(np.array(['o', 'no', 'no', 'bn'])))
print(dt._tree)

# skleann 
clf = DecisionTreeClassifier()
clf.fit(train_data, labels)
print(clf.score(np.array(['o', 'no', 'no', 'bn']), 1))

