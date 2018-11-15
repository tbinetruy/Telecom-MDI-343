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
    scikit_knn.fit(X_train, [y_train.item(i) for i in range(y_train.shape[1])])
    my_knn = KNNClassifier(k).fit(X1, y1)
    for i in range(y_test.shape[1]):
        my_predictions.append(my_knn.predict(X[i]))
        scikit_predictions.append(scikit_knn.predict(X[i]))
    return [my_predictions, [a.item() for a in scikit_predictions]]

my_preds, scikit_preds = test_algo(X1, y1, 3)
print(my_preds)
print(scikit_preds)
