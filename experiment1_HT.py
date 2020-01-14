import csm
import numpy as np
import helper as h
from tqdm import tqdm
import multiprocessing
from csm import OOB, UOB, SampleWeightedMetaEstimator, Dumb, MDET, SEA
from strlearn.evaluators import TestThenTrain
from strlearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    geometric_mean_score_1,
    precision,
    recall,
    specificity
)
from skmultiflow.trees import HoeffdingTree
import sys
from sklearn.base import clone
from sklearn.metrics import accuracy_score

if len(sys.argv) != 2:
    print("PODAJ RS")
    exit()
else:
    random_state = int(sys.argv[1])

print(random_state)

# Select streams and methods
streams = h.toystreams(random_state)

print(len(streams))

oob = OOB()
uob = UOB()
osea = SEA(base_estimator=HoeffdingTree(split_criterion="hellinger") ,oversampled=True)
od = 100
mdet_bac = MDET(optimization_depth=od, metric=balanced_accuracy_score)
mdet_f = MDET(optimization_depth=od, metric=f1_score)

mdet_bac.set_base_clf(HoeffdingTree(split_criterion="hellinger"))
mdet_f.set_base_clf(HoeffdingTree(split_criterion="hellinger"))
oob.set_base_clf(
    SampleWeightedMetaEstimator(HoeffdingTree(split_criterion="hellinger"))
)
uob.set_base_clf(
    SampleWeightedMetaEstimator(HoeffdingTree(split_criterion="hellinger"))
)

clfs = (osea, oob, uob, mdet_bac, mdet_f)

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
        specificity,
        accuracy_score
    ))
    eval.process(
        stream,
        cclfs
    )

    print("Done stream %i/%i" % (i + 1, len(streams)))

    results = eval.scores

    np.save("results/experiment1_HT/%s" % stream, results)


jobs = []
for i, stream_n in enumerate(streams):
    p = multiprocessing.Process(target=worker, args=(i, stream_n))
    jobs.append(p)
    p.start()
