import pandas as pd
import numpy as np
from collections import Counter


class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # list of classes for this model
        self.classes_ = list(set(list(y)))
        # for calculation of P(y)
        self.P_y = Counter(y)
        # Counts how many times the items appear. self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi
        # is the feature value, yj is a specific class label make sure to use self.alpha in the __init__() function
        # as the smoothing factor when calculating P(xi|yj)
        self.P = {}
        # Created a dictionary
        for r in range(X.shape[0]):
            yj = y[r]
            for Xi in range(69):
                # total number of columns is 69
                xi = X.loc[r, Xi]
                if (yj, Xi, xi) in self.P.keys():
                    self.P[yj, Xi, xi] = self.P[yj, Xi, xi] + 1
                else:
                    self.P[yj, Xi, xi] = 1

        # find probability using formula P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha
        for yj, Xi, xi in self.P.keys():
            length = len(set(list(X.loc[:, Xi])))
            self.P[yj, Xi, xi] = (self.P[yj, Xi, xi] + self.alpha) / (self.P_y[yj] + length * self.alpha)

        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(
                    lambda value: self.P[label, key, value] if (label, key, value) in self.P.keys() else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        # convert dataframe to numpy array
        return predictions
