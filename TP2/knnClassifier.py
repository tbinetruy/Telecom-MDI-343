from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import tp_knn_source as tpknn

class KNNClassifier(BaseEstimator, ClassifierMixin):
    n_neightbors = 0
    X = np.matrix([[]])
    y = np.matrix([[]])
    """ Homemade kNN classifier class """
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def distance(self, x, y):
        sum = 0
        for i in range(len(x) + 1):
            sum += (x.item(i) - y.item(i))**2
        return np.sqrt(sum)

    def predict(self, x):
        distances = []
        for i in range(self.X.shape[0]):
            x_i = self.X[i]
            d = self.distance(x_i, x)
            distances.append([i, d])
        sorted_distances = sorted(distances, key=lambda d: d[1])
        k_distances = sorted_distances[:self.n_neighbors]
        k_labels = [self.y.item(k[0]) for k in k_distances]
        k_labels = np.matrix([k_labels])
        return stats.mode(k_labels.T)[0].item()


n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigmas1 = [0.9, 0.9]
sigmas2 = [0.9, 0.9]
X1, y1 = tpknn.rand_bi_gauss(n1, n2, mu1, mu2, sigmas1, sigmas2)
X1 = np.matrix(X1)
y1 = np.matrix(y1).T
x = np.matrix([[-12.093,-10]])

a = KNNClassifier(3).fit(X1, y1).predict(x)
print(a)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X1, [y1.item(i) for i in range(len(y1))])
print(neigh.predict(x))

def split_data(X, y):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(X.shape[0]):
        if i % 2:
            X_train.append(X[i].tolist()[0])
            y_train.append(y.item(i))
        else:
            X_test.append(X[i].tolist()[0])
            y_test.append(y.item(i))
    return [
        np.matrix(X_train),
        np.matrix(y_train),
        np.matrix(X_test),
        np.matrix(y_test)
    ]

def test_algo(X, y, k):
    my_predictions = []
    scikit_predictions = []
    X_train, y_train, X_test, y_test = split_data(X1, y1)
    scikit_knn = KNeighborsClassifier(n_neighbors=k)
    scikit_knn.fit(X_train, np.array(y_train)[0])
    my_knn = KNNClassifier(k).fit(X1, y1)
    for i in range(y_test.shape[1]):
        my_predictions.append(my_knn.predict(X[i]))
        scikit_predictions.append(scikit_knn.predict(X[i]))
    return [my_predictions, [a.item() for a in scikit_predictions]]

my_preds, scikit_preds = test_algo(X1, y1, 3)
print(my_preds)
print(scikit_preds)


from typing import List, Callable

def create_get_weight(h: int) -> Callable[[List[float]], List[float]]:
    def get_weights(distances: List[float]) -> List[float]:
        return [np.exp(-d^2/h) for d in distances]
    return get_weights




from numpy.linalg import inv

class LDAClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        m = np.sum(y[y==1])
        n = y.shape[0]
        pi_plus = m/n
        mu_plus = 1/m * np.sum(X[np.where(y==1),:][0], axis=0).T
        mu_minus = 1/(n-m) * np.sum(X[np.where(y==-1),:][0], axis=0).T


        sig_plus = 1/(m-1) * (X[np.where(y==1),:][0]-mu_plus.T).T@(X[np.where(y==1),:][0]-mu_plus.T)
        sig_minus = 1/(n-m-1) * (X[np.where(y==-1),:][0]-mu_minus.T).T@(X[np.where(y==-1),:][0]-mu_minus.T)

        sig = 1/(n-2) * ((n-1)*sig_plus + (n-m-1)*sig_minus)
        sig_inv = inv(sig)

        t1 = 1/2*mu_plus.T@sig_inv@mu_plus
        t2 = 1/2*mu_minus.T@sig_inv@mu_minus
        self.test = t1 - t2 + np.log(1-pi_plus) - np.log(pi_plus)
        self.sig_inv = sig_inv
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus

        return self

    def predict(self, X):
        y_pred = X@self.sig_inv@(self.mu_plus-self.mu_minus)
        print(y_pred)
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = -1

        return y_pred

    def score(self, y_pred, y):
        print(y_pred.shape)
        return y_pred[y_pred!=y].shape[1]

X = X1
y = y1
my_lda = LDAClassifier().fit(X, y)
y_pred = my_lda.predict(X)
print(my_lda.score(y_pred, y))
