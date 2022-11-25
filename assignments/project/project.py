import pandas as pd
import time
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def cleanup(feat):
    return feat.map(lambda x: str(x).strip("#" + " " + "$"))

def train_test(X):
    return cleanup(X["title"]) + " " \
           + cleanup(X["description"]) + " " \
           + cleanup(X["telecommuting"]) + " " \
           + cleanup(X["has_questions"])


class my_model:
    def __init__(self):
        # defines the self function used in fit and predict
        self.preprocessor = CountVectorizer(stop_words='english', max_df=.7)
        # PassiveAggressiveClassifier gives highest f1 score of 0.766467
        self.clf = PassiveAggressiveClassifier(C=0.1, fit_intercept=True, n_iter_no_change=10, validation_fraction=0.8)


        # LogisticRegression gives lower F1 score than PassiveAggressiveClassifier (F1 score: 0.390625)
        # self.clf = LogisticRegression(solver='liblinear', random_state=0)

        #RandomForestClassifier gives lower F1 score than PassiveAggressiveClassifier (F1 score: 0.743351)
        #self.clf = RandomForestClassifier(n_estimators=100, class_weight= "balanced", random_state=45)

        #SGDClassifier gives lower F1 score than PassiveAggressiveClassifier (F1 score: 0.758242)
        #self.clf = SGDClassifier(class_weight="balanced", max_iter=3000, random_state=45)

        #KNeighborsClassifier(with neighbors = 1) gives lower F1 score than PassiveAggressiveClassifier (F1 score: 0.706468)
        #self.clf = KNeighborsClassifier(n_neighbors=1)

        #SVC gives lower F1 score than PassiveAggressiveClassifier (F1 score: 0.598639)
        #self.clf = SVC(kernel='rbf')

        
    def fit(self, X, y):
        # do not exceed 29 mins
        X_df = train_test(X)
        XX = self.preprocessor.fit_transform(X_df)
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).fit_transform(XX)
        self.clf.fit(X_final, y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X_df = train_test(X)
        XX = self.preprocessor.transform(X_df)
        X_final = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True).fit_transform(XX)
        predictionsOfModel = self.clf.predict(X_final)
        return predictionsOfModel


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    # Train model
    clf = my_model()
    clf.fit(X, y)
    runtime = (time.time() - start) / 60.0
    print(runtime)
    predictions = clf.predict(X)
    print(predictions)

