from csm import SEA, StratifiedBagging, REA, LearnppCDS, LearnppNIE, OUSE, KMeanClustering, rea, TestThenTrain
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
# from strlearn.evaluators import TestThenTrain
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity
)
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
from sklearn.metrics import roc_auc_score

rea = REA(base_classifier=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42), number_of_classifiers=5)
cds = LearnppCDS(base_classifier=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42), number_of_classifiers=5)
nie = LearnppNIE(base_classifier=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42), number_of_classifiers=5)
ouse = OUSE(base_classifier=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42), number_of_classifiers=5)


kmc = KMeanClustering(base_classifier=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42), number_of_classifiers=5)

sea = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42), n_estimators=5, metric=roc_auc_score)


ros_knorau2 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAU2")


stream = StreamGenerator(n_chunks=250, chunk_size=200, random_state=1410, n_drifts=1, weights=[0.9,0.1])
eval = TestThenTrain(metrics=(geometric_mean_score_1))

eval.process(stream, [kmc])
value = np.squeeze(eval.scores[0])
val = gaussian_filter1d(value, sigma=3, mode="nearest")
plt.plot(val)
plt.savefig("zzz")

# print(kmc.new_auc)
# print(kmc.auc_array)
# print(kmc.worst)

# print(eval.scores)
