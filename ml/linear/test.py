import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import LogisticRegression as LogisticRegression_sk
from sklearn.datasets import make_regression
from sklearn.metrics import zero_one_loss

from linear_model import *


# generate data

def random_binary_tensor(shape, sparsity=0.5):
    X = (np.random.rand(*shape) >= (1 - sparsity)).astype(float)
    return X


def random_regression_problem(n_ex, n_in, n_out, intercept=0, std=1, seed=0):
    X, y, coef = make_regression(
        n_samples=n_ex,
        n_features=n_in,
        n_targets=n_out,
        bias=intercept,
        noise=std,
        coef=True,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test, coef


def random_classification_problem(n_ex, n_classes, n_in, seed=0):
    X, y = make_blobs(
        n_samples=n_ex, centers=n_classes, n_features=n_in, random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, y_train, X_test, y_test


def plot_classification():
    # generate data from iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:,:-1], data[:,-1]
    y[y==0] = -1
    num_featues = len(X[0]) # 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    print(X_train)

    # load model
    PE = Perceptron(num_featues)
    PE.fit(X_train, y_train)
    y_pred_pe = PE.predict(X_test)
    score_pe =  PE.score(X_test, y_test)
    print("PE score: ", score_pe)

    LDA = Linear_Discriminant_Analysis(num_featues)
    LDA.fit(X_train[y_train==-1], X_train[y_train==1])
    y_pred_lda = LDA.predict(X_test)
    score_lda =  LDA.score(X_test, y_test)
    print("LDA score: ", score_lda)

    GDA = Gaussian_Discriminant_Analysis(num_featues)
    GDA.fit(X_train, y_train)
    y_pred_gda = GDA.predict(X_test)
    score_gda = GDA.score(X_test, y_test)
    print("GDA score: ", score_gda)

    fig, axes = plt.subplots(1, 4)
    axes[0].scatter(X_test[y_test==-1][:,0], X_test[y_test==-1][:,1], c='blue', alpha=0.5)
    axes[0].scatter(X_test[y_test==1][:,0], X_test[y_test==1][:,1], c='red', alpha=0.5)
    axes[0].set_title("test dataset")

    axes[1].scatter(X_test[y_pred_pe == -1][:,0], X_test[y_pred_pe==-1][:,1], c='blue', alpha=0.5)
    axes[1].scatter(X_test[y_pred_pe == 1][:,0], X_test[y_pred_pe==1][:,1], c='red', alpha=0.5)
    axes[1].set_title("Perception score: %4f" % score_pe)

    axes[2].scatter(X_test[y_pred_lda == -1][:,0], X_test[y_pred_lda==-1][:,1], c='blue', alpha=0.5)
    axes[2].scatter(X_test[y_pred_lda == 1][:,0], X_test[y_pred_lda==1][:,1], c='red', alpha=0.5)
    axes[2].set_title("Fisher score: %4f" % score_lda)

    axes[3].scatter(X_test[y_pred_gda == -1][:,0], X_test[y_pred_gda==-1][:,1], c='blue', alpha=0.5)
    axes[3].scatter(X_test[y_pred_gda == 1][:,0], X_test[y_pred_gda==1][:,1], c='red', alpha=0.5)
    axes[3].set_title("GDA score: %4f" % score_gda)

    fig.set_size_inches(20, 5)
    plt.savefig("plot_linear_cls.png", dpi=300)
    plt.close("all")

def plot_regression():
    np.random.seed(12345)
    n_in = 1
    n_out = 1
    n_ex = 100
    std = 15
    intercept = 10
    X_train, y_train, X_test, y_test, coefs = random_regression_problem(
        n_ex, n_in, n_out, intercept=intercept, std=std, seed=0
    )
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # add some outliers
    x1, x2 = X_train[0] + 0.5, X_train[6] - 0.3
    y1 = np.dot(x1, coefs) + intercept + 25
    y2 = np.dot(x2, coefs) + intercept - 31
    X_train = np.vstack([X_train, np.array([x1, x2])])
    y_train = np.hstack([y_train, [y1[0], y2[0]]])
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Least Squares 
    LS = Least_Squares(fit_intercept=True)
    LS.fit(X_train, y_train)
    y_pred0 = LS.predict(X_test)
    loss0 = np.mean((y_test - y_pred0)**2)/len(X_test)
    print("LS loss: ", loss0)

    # Ridge Regression
    RR = Ridge_Regression(alpha=0.5, fit_intercept=True)
    RR.fit(X_train, y_train)
    y_pred1 = RR.predict(X_test)
    loss1 = np.mean((y_test - y_pred1)**2)/len(X_test)
    print("RR loss: ", loss1)

    # bayesian linear regression unknown prior 
    a = 1e-3
    BLR = BayesianRegression(alpha=a, beta=1.)
    y_preds = []
    for _ in range(100):
        BLR.fit(X_train, y_train)
        y_hat = BLR.predict(X_test) # , return_std=True
        y_preds.append(y_hat.T)
    y_pred2 = np.asarray(y_preds).mean(axis=0)
    loss2 = np.mean((y_test - y_pred2)**2)/len(X_test)
    print("BLR loss: ", loss2)

    # bayesian linear regression known prior 
    a = 1e-3
    m0 = np.mean(X_train)
    S0 = sum([(x-m0)*(x-m0) for x in X_train])/len(X_train)
    BLR_k = BayesianRegression(alpha=a, beta=1., known=True, S0=S0, m0=m0)
    y_preds = []
    for _ in range(100):
        BLR_k.fit(X_train, y_train)
        y_hat = BLR_k.predict(X_test) # , return_std=True
        y_preds.append(y_hat.T)
    y_pred3 = np.asarray(y_preds).mean(axis=0)
    loss3 = np.mean((y_test - y_pred3)**2)/len(X_test)
    print("BLR loss: ", loss3)


    fig, axes = plt.subplots(1, 4)
    axes[0].scatter(X_test, y_test, c='blue', alpha=0.5)
    # axes[0].plot(X_test, y_test, color='green')
    axes[0].plot(X_test, y_pred0, color='red')
    axes[0].set_title("Least Squares")

    axes[1].scatter(X_test, y_test, c='blue', alpha=0.5)
    # axes[1].plot(X_test, y_test, color='green')
    axes[1].plot(X_test, y_pred1, color='red')
    axes[1].set_title("Ridge Regression")

    axes[2].scatter(X_test, y_test, c='blue', alpha=0.5)
    axes[2].plot(X_test, y_pred2, color='orange')
    axes[2].set_title("Bayesian Regression Unknown Var")

    axes[3].scatter(X_test, y_test, c='blue', alpha=0.5)
    axes[3].plot(X_test, y_pred3, color='orange')
    axes[3].set_title("Bayesian Regression Known Var")
    
    fig.set_size_inches(20, 5)
    plt.savefig("plot_logistic.png", dpi=300)
    plt.close("all")

def plot_lr():
    data = np.loadtxt('data/lr_data.csv')
    X = data[:, 0:-1]
    Y = data[:, -1]
    X = np.insert(X, 0, 1, axis=1) # built for b
    _, num_featues = X.shape

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0
    )
    
    print("############# GA #############")
    LR = Logistic_Regression(num_featues)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test.T)
    LR_score = LR.score(X_test.T, y_test)
    print("score: ", LR_score)

    print("############# SGA #############")
    LR_SGA = Logistic_Regression(num_featues)
    LR_SGA.fit(X_train, y_train, method='sga')
    y_pred_sga = LR_SGA.predict(X_test.T)
    LR_SGA_score = LR_SGA.score(X_test.T, y_test)
    print("score: ", LR_SGA_score)

    print("############# mini-SGA #############")
    LR_miniSGA = Logistic_Regression(num_featues)
    LR_miniSGA.fit(X_train, y_train, method='mini-sga')
    y_pred_minisga = LR_miniSGA.predict(X_test.T)
    LR_miniSGA_score = LR_miniSGA.score(X_test.T, y_test)
    print("score: ", LR_miniSGA_score)

    fig, axes = plt.subplots(1, 4)
    axes[0].scatter(X_test[y_test==0][:,1], X_test[y_test==0][:,2], c='blue', alpha=0.5)
    axes[0].scatter(X_test[y_test==1][:,1], X_test[y_test==1][:,2], c='red', alpha=0.5)
    axes[0].set_title("test dataset")

    axes[1].scatter(X_test[y_pred==0][:,1], X_test[y_pred==0][:,2], c='blue', alpha=0.5)
    axes[1].scatter(X_test[y_pred==1][:,1], X_test[y_pred==1][:,2], c='red', alpha=0.5)
    axes[1].set_title("LR result, score: %4f" % (LR_score))

    axes[2].scatter(X_test[y_pred_sga==0][:,1], X_test[y_pred_sga==0][:,2], c='blue', alpha=0.5)
    axes[2].scatter(X_test[y_pred_sga==1][:,1], X_test[y_pred_sga==1][:,2], c='red', alpha=0.5)
    axes[2].set_title("LR sga result, score: %4f" % (LR_SGA_score))

    axes[3].scatter(X_test[y_pred_minisga==0][:,1], X_test[y_pred_minisga==0][:,2], c='blue', alpha=0.5)
    axes[3].scatter(X_test[y_pred_minisga==1][:,1], X_test[y_pred_minisga==1][:,2], c='red', alpha=0.5)
    axes[3].set_title("LR mini-sga result, score: %4f" % (LR_miniSGA_score))

    fig.set_size_inches(20, 5)
    plt.savefig("plot_lr.png", dpi=300)
    plt.close("all")



if __name__ == "__main__":
    # plot_classification()
    # plot_logistic()
    # plot_lr()
    plot_regression()