import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
from math import pi
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import rcParams

from tabulate import tabulate

# Set plot params
rcParams["font.family"] = "monospace"
colors = [(0, 0, 0), (0, 0, 0.9), (0.9, 0, 0), (0, 0, 0), (0, 0, 0.9), (0.9, 0, 0)]
ls = ["--", "--", "--", "-", "-", "-"]
lw = [1, 1, 1, 1, 1, 1]

methods = [
["NON-KNORAU2", "RUS-KNORAU2", "CNN-KNORAU2", "NON-KNORAE2", "RUS-KNORAE2", "CNN-KNORAE2"],
["NON-KNORAU2", "RUS-KNORAU2", "CNN-KNORAU2", "NON-KNORAE2", "RUS-KNORAE2", "CNN-KNORAE2"],
["NON-KNORAU1", "RUS-KNORAU1", "CNN-KNORAU1", "NON-KNORAE2", "RUS-KNORAE2", "CNN-KNORAE2"],
["NON-KNORAU1", "RUS-KNORAU1", "CNN-KNORAU1", "NON-KNORAE1", "RUS-KNORAE1", "CNN-KNORAE1"]
]
len_methods = 6
label_noises = [
    "0.01",
    "0.03",
    "0.05"
]
ln = [a.replace('.','-') for a in label_noises]
distributions = ["0.03"]
dist = [a.replace('.','-') for a in distributions]
drifts = ["gradual", "incremental", "sudden"]
metrics = ["Balanced accuracy", "G-mean", "f1 score", "precision", "recall", "specificity"]
clfs = ["GNB", "HT", "KNN", "SVM"]
seeds = [1994, 1410]
scores = np.load("scores_metrics_22.npy")

# print(scores.shape)

for j, clf in enumerate(clfs):
    print("\n###\n### %s\n###" % (clf))
    for i, metric in enumerate(metrics):
        if metric == "G-mean":
            print("\n---\n--- %s\n---\n" % (metric))
        # clf, seed, drifttype, distr, labelnoise, method, chunk, metric
        sub_scores = scores[j, :, :, :, :, :, :, i]
        # print(sub_scores.shape)
        # seed, drifttype, labelnoise, method, chunk
        reduced_scores = np.mean(sub_scores, axis=2)
        # print(reduced_scores.shape)

        for d, drift in enumerate(drifts):
            counter = 0
            if metric == "G-mean":
                print("\n#######  %s #######\n" % (drift))
            # seed, drifttype, labelnoise, method, chunk
            drift_scores = reduced_scores[:, d, :, :, :]
            mpl = np.zeros((len(seeds)*len(label_noises), len_methods))
            rl = np.zeros((len(seeds)*len(label_noises), len_methods))
            for ln, label_noise in enumerate(label_noises):
                # print("#######  %s #######" % (label_noise))
                # seed, labelnoise, method, chunk
                ln_scores = drift_scores[:, ln, :, :]
                for s, seed in enumerate(seeds):
                    # print("#######  %s #######" % (seed))
                    # seed, method, chunk
                    seed_scores = ln_scores[s, :, :]
                    for m in range(seed_scores.shape[0]):
                        from csm import DriftEvaluator
                        eval_ready_scores = seed_scores[m].reshape(1,199,1)
                        # if metric == "G-mean":
                            # print(eval_ready_scores)
                        drift_evaluator = DriftEvaluator(scores=eval_ready_scores, drift_indices=[99])

                        max_performance_loss = drift_evaluator.get_max_performance_loss()
                        recovery_lengths = drift_evaluator.get_recovery_lengths()
                        mpl[counter, m] = max_performance_loss[0]
                        rl[counter, m] = recovery_lengths[0]
                        # print(max_performance_loss)
                        # print(recovery_lengths)
                    counter += 1
            # całe macierze
                # if metric == "G-mean":
            # print(mpl)
            # print(rl)

            # srednie wyniki
            mean_mpl = np.round(np.mean(mpl, axis=0),3)
            mean_rl = np.round(np.mean(rl, axis=0),3)
            whole = np.concatenate((mean_mpl.reshape(1,len_methods), mean_rl.reshape(1,len_methods)), axis=0)
            # print(mean_mpl)
            # wyswietlenie po bozemu
            # print(tabulate(np.concatenate(([["perf_loss"]], mean_mpl.reshape(1,5)), axis=1), headers=methods, tablefmt="simple"))
            # print(tabulate(np.concatenate(([["reco_leng"]], mean_rl.reshape(1,5)), axis=1), headers=methods, tablefmt="simple"))

            # latex
            # print(tabulate(np.concatenate(([["perf_loss"]], mean_mpl.reshape(1,5)), axis=1), headers=methods, tablefmt="latex_raw"))
            # print(tabulate(np.concatenate(([["reco_leng"]], mean_rl.reshape(1,5)), axis=1), headers=methods, tablefmt="latex_raw"))
            # print(tabulate(np.concatenate(([["perf_loss"],["reco_leng"]], whole), axis=1), headers=methods, tablefmt="simple"))
            if metric == "G-mean":
                print(tabulate(np.concatenate(([["perf_loss"],["reco_leng"]], whole), axis=1), headers=methods[j], tablefmt="latex_raw", floatfmt=".3f"))

            # exit()
