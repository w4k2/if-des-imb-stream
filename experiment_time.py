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
import time
import matplotlib.pyplot as plt

# Select streams and methods
streams = h.timestream(100)

print(len(streams))

ros_knorau2_3 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAU2", n_estimators=3)
ros_knorau2_5 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42, oversampler="CNN"), oversampled="CNN", des="KNORAU2", n_estimators=5)
ros_knorau2_10 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAU2", n_estimators=10)
ros_knorau2_15 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAU2", n_estimators=15)
ros_knorau2_30 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAU2", n_estimators=30)

# cnn_knorau2 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
# ), random_state=42, oversampler="CNN"), oversampled="CNN", des="KNORAU2")
# ros_knorae2 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
# ), random_state=42, oversampler="ROS"), oversampled="ROS", des="KNORAE2")
# cnn_knorae2 = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB(
# ), random_state=42, oversampler="CNN"), oversampled="CNN", des="KNORAE2")

clfs = (
    ros_knorau2_3,
    ros_knorau2_5,
    ros_knorau2_10,
    ros_knorau2_15,
    ros_knorau2_30,
    # cnn_knorau2,
    # ros_knorae2,
    # cnn_knorae2
    )

# n_classifiers worker
# def worker(i, stream_n):
#     time_results = []
#     for i in range(5):
#         streams = h.timestream(250)
#         stream = streams[stream_n]
#         print("Classifier %i" % i)
#         # cclfs = [clone(clf) for clf in clfs]
#         cclfs = clone(clfs[i])
#         print("Starting stream %i/%i" % (i + 1, len(streams)))
#
#         eval = TestThenTrain(metrics=(
#             balanced_accuracy_score,
#             geometric_mean_score_1,
#             f1_score,
#             precision,
#             recall,
#             specificity
#         ))
#         start = time.time()
#         eval.process(
#             stream,
#             cclfs
#         )
#         end = time.time()
#
#         estimated = end - start
#         time_results.append(estimated)
#
#         print("Done stream %i/%i" % (i + 1, len(streams)))
#     print(time_results)
#     plt.plot([3,5,10,15,30], time_results)
#     plt.xticks([3,5,10,15,30], ["3","5","10","15","30"])
#     plt.xlim(3,30)
#     plt.savefig("none_knorae.png")

# chunk_size worker
def worker(i, stream_n):
    chunk_sizes=[100,200,300,400,500,600,700,800]
    time_results = np.zeros((1,8))
    for i in range(1):
        for j in range(8):
            size = chunk_sizes[j]
            streams = h.timestream(size)
            key = list(streams.keys())[0]
            print("Chunk size: ", size)
            stream = streams[key]
            print("Classifier %i" % j)
            # cclfs = [clone(clf) for clf in clfs]
            cclfs = clone(clfs[1])
            print("Starting stream %i/%i" % (j + 1, len(streams)))

            eval = TestThenTrain(metrics=(
                balanced_accuracy_score,
                geometric_mean_score_1,
                f1_score,
                precision,
                recall,
                specificity
            ))
            start = time.time()
            eval.process(
                stream,
                cclfs
            )
            end = time.time()

            estimated = end - start
            time_results[i][j] = estimated

            print("Done stream %i/%i" % (j + 1, len(streams)))
    print(time_results)
    print(np.mean(time_results, axis=0))
    plt.plot([100,200,300,400,500,600,700,800], np.mean(time_results, axis=0))
    plt.xticks([100,200,300,400,500,600,700,800], ["100","200","300","400","500","600","700","800"])
    plt.xlim(100,800)
    plt.savefig("cnn_knorau2_chunks_2.png")

jobs = []
for i, stream_n in enumerate(streams):
    p = multiprocessing.Process(target=worker, args=(i, stream_n))
    jobs.append(p)
    p.start()
