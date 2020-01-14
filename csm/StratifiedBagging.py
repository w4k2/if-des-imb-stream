"""
Stratified Bagging.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn import neighbors
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn.metrics import  geometric_mean_score
import numpy as np
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
import sys, os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from deslib.des import KNORAU
# from desire import DESIRE
from sklearn.neural_network import MLPClassifier

ba = balanced_accuracy_score
f1 = f1_score
gmean = geometric_mean_score


class StratifiedBagging(BaseEstimator, ClassifierMixin):

    def __init__(self, ensemble_size=20, oversampler = "None", des="None", w = 1):
        """Initialization."""
        # self._base_clf = base_estimator
        self.ensemble_size = ensemble_size
        self.oversampler = oversampler
        self.des = des
        self.estimators_ = []
        self.w = w

    def set_base_clf(self, base_clf=GaussianNB()):
        """Establish base classifier."""
        self._base_clf = base_clf

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        minority_X = X[y == 1]
        minority_y = y[y == 1]
        majority_X = X[y == 0]
        majority_y = y[y == 0]

        for i in range(self.ensemble_size):
            self.estimators_.append(base.clone(self._base_clf))

        for n, estimator in enumerate(self.estimators_):
            np.random.seed(42 + n)
            bagXminority = minority_X[np.random.choice(minority_X.shape[0], len(minority_y), replace=True), :]
            bagXmajority = majority_X[np.random.choice(majority_X.shape[0], len(majority_y), replace=True), :]

            bagyminority = np.ones(len(minority_y)).astype('int')
            bagymajority = np.zeros(len(majority_y)).astype('int')

            train_X = np.concatenate((bagXmajority, bagXminority))
            train_y = np.concatenate((bagymajority, bagyminority))

            unique, counts = np.unique(train_y, return_counts=True)

            if self.oversampler == "B2":
                ros = BorderlineSMOTE(random_state=42, kind='borderline-2')
                try:
                    train_X, train_y = ros.fit_resample(train_X, train_y)
                except:
                    pass

            estimator.fit(train_X, train_y)

        # Return the classifier
        return self

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.estimators_])

    def predict(self, X):
        """Hard decision."""

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        if self.des == "KNORAU":
            des = KNORAU(pool_classifiers=self.estimators_, random_state=42)
            des.fit(self.X_, self.y_)
            prediction = des.predict(X)
        elif self.des == "DESIRE":
            des = DESIRE(ensemble=self.estimators_, random_state=42, mode="whole", w=self.w)
            des.fit(self.X_, self.y_)
            prediction = des.predict(X)
        elif self.des == "DESIREC":
            des = DESIRE(ensemble=self.estimators_, random_state=42, mode="correct", w=self.w)
            des.fit(self.X_, self.y_)
            prediction = des.predict(X)
        elif self.des == "DESIREW":
            des = DESIRE(ensemble=self.estimators_, random_state=42, mode="wrong", w=self.w)
            des.fit(self.X_, self.y_)
            prediction = des.predict(X)
        else:
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)

        return prediction

    def predict_proba(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)

        return average_support

    def score(self, X, y):
        return ba(y, self.predict(X))
