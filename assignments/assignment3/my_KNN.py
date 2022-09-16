import pandas as pd
import numpy as np
from collections import Counter
from math import *
from decimal import Decimal

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        # write your code below
        self.X = X
        self.y = y
        return
    def p_root(self, value, root):

        return round(Decimal(value) **
                     Decimal(1/float(root)), 3)

    # Find distance based on metric
    def distance(self, x):
        distances = list()
        for train_row, classLabel in zip(self.X.to_numpy(), self.y):
            distance = 0.0
            z = zip(train_row, x)
            # Pairing train_row and x using zip function.
            if self.metric == "minkowski":
                distance = self.p_root(sum(pow(abs(a - b), self.p) for a, b in z), self.p)

            elif self.metric == "euclidean":
                distance = sqrt(sum(pow(a - b, 2) for a, b in z))


            elif self.metric == "manhattan":
                distance = sum(abs(a - b) for a, b in z)


            elif self.metric == "cosine":
                distance = np.dot(train_row, x) / (np.sqrt(np.dot(train_row, x)) * np.sqrt(np.dot(train_row, x)))

            distances.append((classLabel, distance))

        return distances

    # Locate the most similar neighbors
    def k_neighbors(self, x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors)
        distances = self.distance(x)
        distances.sort(key=lambda tup: tup[1], reverse=True)
        #Arranging in decending order
        neighbors = list()
        for i in range(self.n_neighbors):
            neighbors.append(distances[i][0])

        return Counter(neighbors)

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]

        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = []

        for x in X[self.X.columns].to_numpy():
            neighbors = self.k_neighbors(x)
            probs.append({key: neighbors[key] / float(self.n_neighbors) for key in self.classes_})

        probs = pd.DataFrame(probs, columns=self.classes_)

        return probs



