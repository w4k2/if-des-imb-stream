import csm
import numpy as np
import helper as h
from tqdm import tqdm
import multiprocessing
from csm import OOB, UOB, SampleWeightedMetaEstimator, Dumb, MDET, SEA, StratifiedBagging
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

if len(sys.argv) != 2:
    print("PODAJ RS")
    exit()
else:
    random_state = int(sys.argv[1])

print(random_state)

# Select streams and methods
streams = h.toystreams(random_state)

print(len(streams))

none_knorau1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(
    probability=True, random_state=42), random_state=42, oversampler="None"), oversampled="None", des="KNORAU1")
rus_knorau1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(
    probability=True, random_state=42), random_state=42, oversampler="RUS"), oversampled="RUS", des="KNORAU1")
cnn_knorau1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(
    probability=True, random_state=42), random_state=42, oversampler="CNN"), oversampled="CNN", des="KNORAU1")
none_knorae1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(
    probability=True, random_state=42), random_state=42, oversampler="None"), oversampled="None", des="KNORAE1")
rus_knorae1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(
    probability=True, random_state=42), random_state=42, oversampler="RUS"), oversampled="RUS", des="KNORAE1")
cnn_knorae1 = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(
    probability=True, random_state=42), random_state=42, oversampler = "CNN"), oversampled="CNN" ,des="KNORAE1")

clfs = (none_knorau1, rus_knorau1, cnn_knorau1, none_knorae1, rus_knorae1, cnn_knorae1)

# Define worker
def worker(i, stream_n):
    stream = streams[stream_n]
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

    np.save("results/experiment22_SVM/%s" % stream, results)


jobs = []
for i, stream_n in enumerate(streams):
    p = multiprocessing.Process(target=worker, args=(i, stream_n))
    jobs.append(p)
    p.start()
