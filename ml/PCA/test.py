import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA as PCA_sk
from pca import *

def pca_analysis():
    # generate data from iris
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['label'] = boston.target
    data = np.array(df.iloc[:100]) # [0, 1, -1]
    X, y = data[:,:-1], data[:,-1]
    y[y==0] = -1
    num_featues = len(X[0]) # 2
    # print("num_features: ", num_featues)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    # print(X_train.shape)
    PCA = pca()
    PCA.fit(X_train)
    pca_result = PCA.fit_transfer(X_train)
    print("pca: ", pca_result)
    # print("pca eig value: ", PCA.eigValInd)
    # print("pca eig vector: ", PCA.redEigVects)

    # sklearn pca 
    pca_sk = PCA_sk(n_components=3)
    pca_sk.fit(X_train)
    # print(pca_sk.explained_variance_)
    # print(pca_sk.singular_values_)
    principalComponents = pca_sk.fit_transform(X_train)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2',
                        'principal component 3'])
    print("sklearn pca: ", principalDf)

if __name__ == "__main__":
    pca_analysis()