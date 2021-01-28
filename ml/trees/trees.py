import numpy as np
from sklearn.metrics import accuracy_score
from math import log


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label:': self.label,
            'feature_name': self.feature_name,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature_name]].predict(features)

class DecisionTree():
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self._tree = {}

    def bincount2D(self, a, axis=0):
        N = a.max() + 1
        if axis == 0:
            a_offs = a + np.arange(a.shape[axis])[:,None]*N
            return np.bincount(a_offs.ravel(), minlength=a.shape[axis]*N).reshape(-1,N)
        elif axis == 1:
            a_offs = a.T + np.arange(a.shape[axis])[:,None]*N
            return np.bincount(a_offs.ravel(), minlength=a.shape[axis]*N).reshape(-1,N).T
        else:
            print("only support 2D")

    def calc_ent(self, y):
        # y [1, -1]
        # p(x)*log2(p(x))
        y = np.array([int(x) for x in y]).reshape(-1,1)
        hist = np.bincount(y.flatten())
        hist = hist[hist!=0]
        p = hist/len(y)
        return -np.sum([px * np.log2(px) for px in p])

    def calc_cond_ent(self, X, y):
        # print(X.shape, y.shape)
        X = X[:,np.newaxis]
        y = y[:,np.newaxis]
        # print(X.shape, y.shape)
        dataset = np.concatenate((X, y), axis=1)
        features = np.unique(X)
        cond_ent = 0
        for f in features:
            sub_dataset = dataset[np.argwhere(X[:,0]==f).squeeze(axis=1), :]
            cond_ent += sub_dataset.shape[0]/X.shape[0] * self.calc_ent(sub_dataset[:,-1])
        return cond_ent

    def calc_info_gain(self, X, y):
        return self.calc_ent(y) - self.calc_cond_ent(X, y)

    def select_feature(self, X, y):
        x_feats = X.shape[1]
        best_feature = []
        for i in range(x_feats):
            info_gain = self.calc_info_gain(X[:,i], y)
            best_feature.append((i, info_gain))
        return max(best_feature, key=lambda x: x[-1])

    def fit(self, X, y, catgory):
        # fit a binary dt
        # X: [M, N], M number of samples, N: feature number
        # y: [M, 1]
        self.n_classes = len(np.unique(y))
        self._tree = self.grow(X, y, catgory)
        return self._tree

    def grow(self, X, y, category):
        # if all labels are the same, reture a leaf
        if len(np.unique(y)) == 1:
            return Node(root=True, label = y[0])
        
        # if dataset is empty
        if len(X) == 0:
            argmax = np.argmax(np.bincount(y.flatten()))
            return Node(root=True, label = y[argmax])

        # if self.depth >= self.max_depth:
        #     v = np.bincount(y, minlength=self.n_classes) / len(y)
        #     return Leaf(v)

        max_feat_ind, max_info_gain = self.select_feature(X, y)
        if max_info_gain < self.epsilon:
            # return a singal node tree
            y = np.array([int(x) for x in y]).reshape(-1,1)
            argmax = np.argmax(np.bincount(y.flatten()))
            return Node(root=True, label = y[argmax])

        node_tree = Node(root=False, feature_name=max_feat_ind, 
            feature=X[:, max_feat_ind]) # X[:, max_feat_ind]
        child_features = np.unique(X[:, max_feat_ind])
        sub_category = [category[i] for i in range(len(category)) if i != max_feat_ind]
        for child_feature in child_features:
            sub_X = X[X[:,max_feat_ind]==child_feature]
            sub_y = y[X[:,max_feat_ind]==child_feature]
            sub_X = sub_X[:, [i for i in range(sub_X.shape[1]) if i != max_feat_ind]]
            sub_tree = self.grow(sub_X, sub_y, sub_category)
            node_tree.add_node(child_feature, sub_tree)
        return node_tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

    def score(self, X, y):
        if len(X) == 1:
            y_pred = self.predict(X)
        else:
            y_pred = []
            for x in X:
                print(x)
                y_pred.append(self.predict(x))
            y_pred = np.array(y_pred).reshape(-1,1)
        return accuracy_score(y, y_pred)


"""
class CART():
    # classification and regression tree
    def __init__(self, epsilon, classifier):
        self.epsilon = epsilon
        self._tree = {}
        self.classifier = classifier

    def cal_ent(self, y):
        # y [1, -1]
        # p(x)*log2(p(x))
        y = np.array([int(x) for x in y]).reshape(-1,1)
        hist = np.bincount(y.flatten())
        hist = hist[hist!=0]
        p = hist/len(y)
        return -np.sum([px * np.log2(px) for px in p])

    def cal_cond_ent(self, X, y):
        X = X[:,np.newaxis]
        dataset = np.concatenate((X, y), axis=1)
        features = np.unique(X)
        cond_ent = 0
        for f in features:
            sub_dataset = dataset[np.argwhere(X[:,0]==f).squeeze(axis=1), :]
            cond_ent += sub_dataset.shape[0]/X.shape[0] * self.cal_ent(sub_dataset[:,-1])
        return cond_ent

    def cal_info_gain(self, ent, cond_ent):
        return ent - cond_ent

    def gini(self, y):
        hist = np.bincount(y)
        N = np.sum(hist)
        return 1 - sum([(i / N) ** 2 for i in hist])

    def fit(self, X, y, category):
        # fit a binary dt
        # X: [M, N], M number of samples, N: feature number
        # y: [M, 1]
        self.n_classes = len(np.unique(y))
        self._tree = self.grow(X, y, category)
        return self._tree

    def grow(self, X, y, category):
"""      


        
        

                

            

        



    

