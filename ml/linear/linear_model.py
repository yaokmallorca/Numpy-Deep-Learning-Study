import numpy as np
import math
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

class Least_Squares():
    """
    orginal least squares:
    f(x) = w^T*X = X*beta
    beta = (x^Tx)^{-1}X^TY 
    """
    def __init__(self, fit_intercept=True):
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        return np.dot(X, self.beta)

class Ridge_Regression():
    """
    f(x) = w^T*X = X*beta
    beta = (x^Tx+alpha*I)^{-1}X^TY 
    """
    def __init__(self, alpha, fit_intercept=True):
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        A = self.alpha * np.eye(X.shape[1])
        self.beta = np.dot(np.dot(np.linalg.inv((np.dot(X.T, X) + A)), X.T), y)

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        # print(self.beta.shape, X.shape)
        return np.dot(X, self.beta)

class Perceptron():
    """
    f(x) = sign(wx+b)
    SGD: 
    w = w + lr*yi*xi
    b = b + lr*yi
    """
    def __init__(self, num_feature, lr=0.01):
        self.w = np.ones(num_feature, dtype=np.float32)
        self.b = 0
        self.lr = lr

    def sign(self, x):
        y = np.dot(self.w, x) + self.b
        return y

    def fit(self, X, y):
        is_wrong = False
        while not is_wrong:
            wrong_cnt = 0
            for d in range(len(X)):
                x_train = X[d]
                y_train = y[d]
                if y_train * self.sign(x_train) <= 0:
                    self.w = self.w + self.lr*np.dot(y_train,x_train)
                    self.b = self.b + self.lr*y_train
                    wrong_cnt += 1
            if wrong_cnt == 0:
                is_wrong = True
        return

    def predict(self, x):
        pred = -(np.dot(self.w, x.T) + self.b)/np.linalg.norm(self.w)
        pred_label = np.zeros(pred.size)
        pred_label[pred>0] = -1
        pred_label[pred<=0] = 1
        return pred_label

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

# Fisher
class Linear_Discriminant_Analysis():  
    """
    w = S_w^{-1}*(u0 - u1)
    """
    def __init__(self, num_features):
        self.w = np.zeros(num_features)
        self.u1 = np.zeros(num_features)
        self.u2 = np.zeros(num_features)

    def fit(self, X1, X2):
        self.u1 = np.mean(X1, axis=0)
        self.u2 = np.mean(X2, axis=0)
        cov1 = np.dot((X1 - self.u1).T, (X1 - self.u1))
        cov2 = np.dot((X2 - self.u2).T, (X2 - self.u2))
        s_w = cov1 + cov2
        self.w = np.dot(np.linalg.inv(s_w), (self.u1 - self.u2).reshape((len(self.u1), 1)))

    def predict(self, X):
        center1 = np.dot(self.w.T, self.u1)
        center2 = np.dot(self.w.T, self.u2)
        pos = np.dot(self.w.T, X.T)
        pred = (np.int8(abs(pos - center1) > abs(pos - center2))).squeeze()
        pred_label = np.zeros(pred.size)
        pred_label[pred==1] = 1
        pred_label[pred==0] = -1
        return pred_label

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class Logistic_Regression:
    def __init__(self, num_features, penalty='l2', gamma=0.1):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.w = np.random.rand(num_features)
        self.gamma = gamma
        self.penalty = penalty

    def sigmoid(self, X):
        return 1./(1. + np.exp(-np.dot(self.w.T, X)))

    def NLL(self, X, y, y_pred):
        order = 2 if self.penalty == 'l2' else 1
        nll = np.sum(-np.log(y_pred) - np.log(1.-y_pred))
        penalty = 0.5 * self.gamma * np.linalg.norm(self.w, ord=order)**2
        return (penalty + nll)/float(y.size)
    
    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e2, method='ga', batch_size=10):
        if method == 'ga':
            self.gradient_ascent(X, y, lr, tol, max_iter)
        elif method == 'sga':
            self.stochastic_gradient_ascent(X, y, lr, tol, max_iter)
        elif method == 'mini-sga':
            self.minibatch_stochastic_gradient_ascent(X, y, lr, tol, max_iter, batch_size)
        else:
            print("please use correct method.")

    def gradient_ascent(self, X, y, lr, tol, max_iter):
        for i in range(int(max_iter)):
            y_pred = self.sigmoid(X.T)
            loss = self.NLL(X, y, y_pred)
            if loss < tol or np.isnan(loss) or np.isinf(loss):
                print("[{}] loss: {}".format(i, loss))
                return
            self.w = self.w + lr*np.dot((y-y_pred), X)
            print("[{}] loss: {}".format(i, loss))

    def stochastic_gradient_ascent(self, X, y, lr, tol, max_iter):
        for i in range(int(max_iter)):
            X_stoc, y_stoc = shuffle(X, y)
            loss = []
            for X_sample, y_sample in zip(X_stoc, y_stoc):
                y_pred = self.sigmoid(X_sample)
                nll_loss = self.NLL(X_sample, y_sample, y_pred)
                self.w = self.w + lr*np.dot((y_sample-y_pred), X_sample)
                loss.append(nll_loss)
            if np.isnan(sum(loss)) or np.isinf(sum(loss)):
                print("[{}] loss: {}".format(i, np.mean(loss)))
                return 
            print("[{}] loss: {}".format(i, np.mean(loss)))

    def minibatch_stochastic_gradient_ascent(self, X, y, lr, tol, max_iter, batch_size):
        for ind in range(int(max_iter)):
            X_stoc, y_stoc = shuffle(X, y)
            loss = []
            iters = int(y.size/batch_size)
            for i in range(iters):
                X_sample = X_stoc[i*batch_size:(i+1)*batch_size]
                y_sample = y_stoc[i*batch_size:(i+1)*batch_size]
                y_pred = self.sigmoid(X_sample.T)
                nll_loss = self.NLL(X_sample, y_sample, y_pred)
                self.w = self.w + lr*np.dot((y_sample-y_pred), X_sample)
                loss.append(nll_loss)
            if np.isnan(sum(loss)) or np.isinf(sum(loss)):
                print("[{}] loss: {}".format(ind, np.mean(loss)))
                return 
            print("[{}] loss: {}".format(ind, np.mean(loss)))

    def predict(self, X):
        return np.int8(self.sigmoid(X)>=0.5)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class SoftmaxRregression():
    def __init__(self, lr=0.01, num_features):
        self.lr = lr

    def softmax(self, X):
        exp_x = np.exp(np.dot(self.w.T, X))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def label_one_hot(self, y):
        n_samples = len(y)
        n_classes = len(np.unique(y))
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot

    def cross_entropy(self, X, y_t, y_p):
        # loss = -(1/m)*sum_{i=n_samples}sum_{k=n_classes}(y_k^i*log(p_k^i))
        loss = -(1/len(X))*np.sum(y_t * np.log(y_p))
        return loss

    def fit(self, X, y, max_iters=100):
        self.w = np.random.rand(n_classses, n_features)
        n_features = X.shape[1]
        n_classses = y.shape[1]
        self.b = np.zeros((1, n_classses))
        y_one_hot = self.label_one_hot(y)
        losses = []
        for i in range(max_iters):
            probs = self.softmax(X)
            y_pred = np.argmax(probs, axis=1)[:, np.newaxis]
            loss = self.cross_entropy(y_one_hot, y_pred)
            # update w & b
            self.w = self.w - self.lr*(np.dot(X.T, (probs-y_one_hot))/X.shape[0])
            self.b = self.b - self.lr*(np.sum(probs-y_one_hot, axis=0)/X.shape[0])
            if i % 100 == 0:
                print(f'[i], loss: {np.round(loss, 4)}')

    def predict(self, X):
        probs = self.softmax(X)
        return np.argmax(probs, axis=1)[:, np.newaxis]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class Gaussian_Discriminant_Analysis():
    # compute phi, u0, u1 and std
    def __init__(self, num_features):
        self.phi = None
        self.mu_negetive = None
        self.mu_positive = None
        self.sigma = None

    def fit(self, X, y):
        X_1 = X[y==1]
        X_0 = X[y==-1]
        self.phi = float(len(y[y==1])/y.size)
        self.mu_negetive = np.mean(X_0, axis=0)
        self.mu_positive = np.mean(X_1, axis=0)
        self.sigma = (np.cov(X_0.T) * len(X_0) + np.cov(X_1.T) * len(X_1))/float(len(X))

    def Gaussian2d(self, X, mean):
        dim = np.shape(self.sigma)[0]
        covdet = np.linalg.det(self.sigma + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(self.sigma + np.eye(dim) * 0.001)
        xdiff = (X - mean).reshape((1, dim))
        prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
               np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def predict(self, test_data):
        predict_label = []
        for data in test_data:
            positive_pro = self.Gaussian2d(data, self.mu_positive)
            negetive_pro = self.Gaussian2d(data, self.mu_negetive)
            if positive_pro >= negetive_pro:
                predict_label.append(1)
            else:
                predict_label.append(-1)
        return np.int8(predict_label)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class BayesianRegression():
    """
    Bayesian regression model
    mn = Sn*(S0^{-1}*m0 + beta*X*y)
    Sn^{-1} = S0^{-1} + beta*X.T*X
    """
    def __init__(self, alpha=1., beta=1., known=False, S0=None, m0=None):
        self.alpha = alpha
        self.beta = beta
        if known == True:
            self.Sn = S0
            self.mn = m0
        else:
            self.Sn = np.array([])
            self.mn = np.array([])

    def get_prior(self, num_features):
        if self.Sn.size == 0 and self.mn.size == 0:
            return np.zeros((num_features, 1)), self.alpha*np.eye(num_features)
        else:
            return self.mn, self.Sn

    def fit(self, X, y):
        m0, S0 = self.get_prior(len(X[0]))
        try: 
            S0_inv = np.linalg.inv(S0)
        except:
            S0_inv = np.linalg.inv(S0[:,np.newaxis])
        self.Sn = S0_inv + self.beta * np.dot(X.T, X)
        self.mn = np.dot(np.linalg.inv(self.Sn), (np.dot(S0_inv, m0) + self.beta*np.dot(X.T, y)))
        self.w_cov = np.linalg.inv(self.Sn)

    def predict(self, X, sample_size=None, return_std=False):
        # given x*, noise-free
        # p(f(x*)) ~ N(w.T@X*, X*.T @ w_cov @ X*)
        # with noise N(0, sigma^2)
        # p(f(x*)) ~ N(w.T@X*, X*.T @ w_cov @ X* + sigma^2)
        y = np.dot(self.mn, X.T)
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov.T * X, axis=1) # 
            y_std = np.sqrt(y_var)
            return y, y_std
        return y


class BaysianLogisticRegression():
    """
    Newton method:
    w_{k+1} = w_{k} - H_k^{-1}*f(x)_grad
    H: hessian matrix
    w ~ Gaussian(0, alpha^(-1)I)
    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """
    def __init__(self, alpha=1.):
        self.alpha = alpha
    
    def sigmoid(self, X):
        return 1. / (1. + np.exp(-X))

    def fit(self, X, y, max_iter):
        w = np.zeros((X.shape[1],1))
        eye = np.eye((X.shape[1],1))

        self.w_mean = np.copy(w)
        self.w_precision = self.alpha * eye
        for i in range(max_iter):
            w_prev = np.copy(w)
            pred = self.sigmoid(np.dot(w, X.T))
            grad = np.dot(X.T, (y-pred)) + np.dot(self.w_precision, (w - self.w_mean))
            hessian = np.dot((X.T * pred * (1 - y)), X) + self.w_precision

            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w_mean = w
        self.w_precision = hessian

    def predict(self, X):
        mu_a = X @ self.w_mean
        var_a = np.sum(np.linalg.solve(self.w_precision, X.T).T * X, axis=1)
        return self.sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))


        

                
    
