"""Chunk based ensemble."""

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from deslib.des import KNORAE
from utils import KNORAU
from strlearn.metrics import balanced_accuracy_score


class SEA(ClassifierMixin, BaseEnsemble):
    """
    Streaming Ensemble Algorithm.

    Ensemble classifier composed of estimators trained on the fixed
    number of previously seen data chunks, prunning the worst one in the pool.

    Parameters
    ----------

    n_estimators : integer, optional (default=10)
        The maximum number of estimators trained using consecutive data chunks
        and maintained in the ensemble.
    metric : function, optional (default=accuracy_score)
        The metric used to prune the worst classifier in the pool.

    Attributes
    ----------
    ensemble_ : list of classifiers
        The collection of fitted sub-estimators.
    classes_ : array-like, shape (n_classes, )
        The class labels.

    Examples
    --------
    >>> import strlearn as sl
    >>> stream = sl.streams.StreamGenerator()
    >>> clf = sl.ensembles.SEA()
    >>> evaluator = sl.evaluators.TestThenTrainEvaluator()
    >>> evaluator.process(clf, stream)
    >>> print(evaluator.scores_)
    ...
    [[0.92       0.91879699 0.91848191 0.91879699 0.92523364]
    [0.945      0.94648779 0.94624912 0.94648779 0.94240838]
    [0.925      0.92364329 0.92360881 0.92364329 0.91017964]
    ...
    [0.925      0.92427885 0.924103   0.92427885 0.92890995]
    [0.89       0.89016179 0.89015879 0.89016179 0.88297872]
    [0.935      0.93569212 0.93540766 0.93569212 0.93467337]]
    """

    def __init__(self, base_estimator=None, n_estimators=5, metric=balanced_accuracy_score, oversampled="None", des="None"):
        """Initialization."""
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.metric = metric
        self.oversampled = oversampled
        self.des = des

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []
            self.ensemble_base_ = []

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")

        self.X_, self.y_ = X, y
        if self.oversampled == "None":
            self.dsel_X_, self.dsel_y_ =  self.X_, self.y_
        elif self.oversampled == "ROS":
            ros = RandomOverSampler(random_state=42)
            try:
                self.dsel_X_, self.dsel_y_ = ros.fit_resample(self.X_, self.y_)
            except:
                self.dsel_X_, self.dsel_y_ = self.X_, self.y_
        elif self.oversampled == "B2":
            b2 = BorderlineSMOTE(random_state=42, kind='borderline-2')
            try:
                self.dsel_X_, self.dsel_y_ = b2.fit_resample(self.X_, self.y_)
            except:
                self.dsel_X_, self.dsel_y_ = self.X_, self.y_
        elif self.oversampled == "RUS":
            rus = RandomUnderSampler(random_state=42)
            try:
                self.dsel_X_, self.dsel_y_ = rus.fit_resample(self.X_, self.y_)
                # _, ys_counter = np.unique(self.dsel_y_, return_counts=True)

                # if np.sum(ys_counter) < 9:
                    # rus = RandomUnderSampler(random_state=42, sampling_strategy={0:(9-ys_counter[1]), 1:ys_counter[1]})
                    # self.dsel_X_, self.dsel_y_ = rus.fit_resample(self.X_, self.y_)
            except:
                self.dsel_X_, self.dsel_y_ = self.X_, self.y_
        elif self.oversampled == "CNN":
            cnn = CondensedNearestNeighbour(random_state=42)
            try:
                self.dsel_X_, self.dsel_y_ = cnn.fit_resample(self.X_, self.y_)
            except:
                self.dsel_X_, self.dsel_y_ = self.X_, self.y_

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Append new estimator
        self.candidate_ = clone(self.base_estimator).fit(self.X_, self.y_)
        self.ensemble_.append(self.candidate_)
        self.ensemble_base_.extend(self.candidate_.estimators_)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            self.prune_index_ = np.argmin(
                [self.metric(y, clf.predict(X)) for clf in self.ensemble_])
            # print(self.prune_index_)
            del self.ensemble_[self.prune_index_]
            a = (((self.prune_index_ + 1) * 10) - 10)
            b = (((self.prune_index_ + 1) * 10))
            del self.ensemble_base_[a:b]
            # print(a, ":", b)

        return self


    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

# 0.3, 0.7


    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        if self.des == "None":
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
        elif self.des == "KNORAU1":
            des = KNORAU(ensemble=self.ensemble_)
            des.fit(self.dsel_X_, self.dsel_y_)
            prediction = des.predict(X)
        elif self.des == "KNORAU2":
            des = KNORAU(ensemble=self.ensemble_base_)
            des.fit(self.dsel_X_, self.dsel_y_)
            prediction = des.predict(X)
        elif self.des == "KNORAE1":
            des = KNORAE(pool_classifiers=self.ensemble_, random_state=42)
            des.fit(self.dsel_X_, self.dsel_y_)
            prediction = des.predict(X)
        elif self.des == "KNORAE2":
            des = KNORAE(pool_classifiers=self.ensemble_base_, random_state=42)
            des.fit(self.dsel_X_, self.dsel_y_)
            prediction = des.predict(X)

        # Return prediction
        return self.classes_[prediction]
