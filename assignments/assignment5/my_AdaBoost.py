import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace
import math
class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50, learning_rate=1):
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # If one base estimator predicts perfectly,
            # Use that base estimator only
            if (error <= 0.5):
                alpha = self.learning_rate*math.log((1.0 - error) / error) + np.log(k-1)
            else:
                alpha = self.learning_rate*math.log((1.0-error)/error)
            self.alpha.append(alpha)

            # Compute alpha for estimator i (don't forget to use k for multi-class)


            # Update wi
            w1 = []
            for w, isError in zip(w, diffs):
                if isError:
                    w1.append(w*np.exp(alpha))
                else:
                    w1.append(w)
            w =np.array(w1)
            w /= np.sum(w)


        # Normalize alpha
        self.alpha = self.alpha / np.sum(self.alpha)
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # Note that len(self.estimators) can sometimes be different from self.n_estimators
        # write your code below
        probs = []
        list_predictions = []
        for j in range(self.n_estimators):
            predictions = self.estimators[j].predict(X)
            list_predictions.append(predictions)

        dflist_predictions = pd.DataFrame(list_predictions)
        for col in range(dflist_predictions.shape[1]):

            pClass = dict()
            for name in self.classes_:
                pClass[name] = 0

            for row in range(dflist_predictions.shape[0]):
                classname = dflist_predictions.iloc[row, col]
                pClass[classname] = pClass[classname] + self.alpha[row]


            probs.append({name:pClass[name] for name in self.classes_})
            # Calculate probs for each label
            "write your own code"



        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs





