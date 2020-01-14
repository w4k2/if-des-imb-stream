"""
Minority Driven Ensemble.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from strlearn.metrics import balanced_accuracy_score
import numpy as np


class MDET(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        ensemble_size=3,
        alpha=0.05,
        optimization_depth=50,
        t_strategy="auto",
        metric=balanced_accuracy_score,
    ):
        """Initialization."""
        self.ensemble_size = ensemble_size
        self.alpha = alpha
        self.optimization_depth = optimization_depth
        self.opt_quants = np.linspace(0, 1, self.optimization_depth)
        self.t_strategy = t_strategy
        self.metric = metric

    def set_base_clf(self, base_clf=GaussianNB()):
        """Establish base classifier."""
        self._base_clf = base_clf

    # Fitting
    def fit(self, X, y):
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
        X, y = check_X_y(X, y)

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes
            self.ensemble_ = []
            self.weights_ = []
            self.tresholds_ = []

        self.X_, self.y_ = X, y

        # Testing all models
        scores = np.array([self.metric(y, clf.predict(X)) for clf in self.ensemble_])

        # Pruning
        self.prune(scores)

        # Preparing and training new candidate
        candidate_clf = base.clone(self._base_clf).fit(self.X_, self.y_)

        # Checking tresholds
        if self.t_strategy == "auto":
            probas = candidate_clf.predict_proba(self.X_)[:, 0]
            treshold = self.opt_quants[
                np.argmax([self.metric(self.y_, probas < t) for t in self.opt_quants])
            ]
        else:
            treshold = self.t_strategy

        self.ensemble_.append(candidate_clf)
        self.tresholds_.append(treshold)

    # Pruning
    def prune(self, scores):
        if len(self.ensemble_) > 1:
            alpha_good = scores > (0.5 + self.alpha)
            self.ensemble_ = [self.ensemble_[i] for i in np.where(alpha_good)[0]]
            self.tresholds_ = [self.tresholds_[i] for i in np.where(alpha_good)[0]]

        if len(self.ensemble_) > self.ensemble_size - 1:
            worst = np.argmin(scores)
            del self.ensemble_[worst]
            del self.tresholds_[worst]

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        minority_support = esm[:, :, 0]

        base_predictions = np.zeros((len(self.ensemble_), X.shape[0]))
        for i in range(len(self.ensemble_)):
            base_predictions[i] = minority_support[i] < self.tresholds_[i]

        prediction = np.min(base_predictions, axis=0)

        return prediction

    def score(self, X, y):
        return measure(y, self.predict(X))
