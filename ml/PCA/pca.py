import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

class pca():
    """
    input: X = (x1, x2, ..., xn)^T 
    S: invariace matrix
    pca: S u_1 = \lambda u_1
    """
    def __init__(self):
        self.mu = None
        self.eigValInd = None
        self.redEigVects = None

    # x: {num_sample, num_features}
    # 
    def fit(self, X, n_components=3):
        self.mu = X.mean(axis=0)
        X_u = X - self.mu
        covM = np.cov(X_u,rowvar=0)
        eigValues , eigVectors = np.linalg.eig(covM)
        # eigValInd = np.argsort(eigValues)
        self.eigValInd = eigValues[:n_components]
        self.redEigVects = eigVectors[:,:n_components]

    def fit_transfer(self, X):
        X_u = X - self.mu
        X_adjust = np.dot(X_u, self.redEigVects)
        return X_adjust


