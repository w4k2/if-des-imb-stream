import csm
import numpy as np
import helper as h
from tqdm import tqdm
import multiprocessing
from csm import OOB, UOB, SampleWeightedMetaEstimator, Dumb, MDET, SEA, StratifiedBagging, OnlineBagging
from strlearn.evaluators import TestThenTrain
from sklearn.naive_bayes import GaussianNB
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity
)
import sys
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.trees import HoeffdingTree

# Select streams and methods
streams = h.realstreams()
print(len(streams))

ob = OnlineBagging(n_estimators=20, base_estimator=HoeffdingTree(
    split_criterion='hellinger'))
oob = OOB(n_estimators=20, base_estimator=HoeffdingTree(
    split_criterion='hellinger'))
uob = UOB(n_estimators=20, base_estimator=HoeffdingTree(
    split_criterion='hellinger'))
ros_knorau2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTree(
    split_criterion='hellinger'), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAU2")
cnn_knorau2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTree(
    split_criterion='hellinger'), random_state=42, oversampler="CNN"), oversampled="CNN", des="KNORAU2")
ros_knorae2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTree(
    split_criterion='hellinger'), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAE2")
cnn_knorae2 = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTree(
    split_criterion='hellinger'), random_state=42, oversampler = "CNN"), oversampled="CNN" ,des="KNORAE2")

clfs = (ob, oob, uob, ros_knorau2, cnn_knorau2, ros_knorae2, cnn_knorae2)

# Define worker
def worker(i, stream_n):
    stream = streams[stream_n]
    key = list(streams.keys())[i]

    cclfs = [clone(clf) for clf in clfs]

    print("Starting stream %i/%i" % (i + 1, len(streams)))

    eval = TestThenTrain(metrics=(
        balanced_accuracy_score,
        geometric_mean_score_1,
        f1_score,
        precision,
        recall,
        specificity
    ))
    eval.process(
        stream,
        cclfs
    )

    print("Done stream %i/%i" % (i + 1, len(streams)))

    results = eval.scores
    # print(eval.scores)

    np.save("results/experiment4_HT/%s" % key, results)


jobs = []
for i, stream_n in enumerate(streams):
    p = multiprocessing.Process(target=worker, args=(i, stream_n))
    jobs.append(p)
    p.start()
