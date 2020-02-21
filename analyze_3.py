import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
from math import pi
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib import rcParams

# Set plot params
rcParams["font.family"] = "monospace"
colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0.9), (0, 0, 0.9), (0.9, 0, 0), (0.9, 0, 0)]
ls = ["-", "--", ":", "-", "--", "-", "--"]
lw = [1, 1, 1, 1, 1, 1, 1]

methods = ["OB", "OOB", "UOB", "ROS-KNORAU2", "CNN-KNORAU2", "ROS-KNORAE2", "CNN-KNORAE2"]
label_noises = [
    "0.01",
    "0.03",
    "0.05"
]
ln = [a.replace('.','-') for a in label_noises]
distributions = ["0.05", "0.10", "0.15"]
dist = [a.replace('.','-') for a in distributions]
drifts = ["gradual", "incremental", "sudden"]
metrics = ["Balanced accuracy", "G-mean", "f1 score", "precision", "recall", "specificity"]
clfs = ["GNB", "HT"]

scores = np.load("scores_3.npy")

# print(scores.shape)

def plot_runs(
    clfs, metrics, selected_scores, methods, mean_scores, dependency, what
):
    fig = plt.figure(figsize=(4.5, 3))
    ax = plt.axes()
    for z, (value, label, mean) in enumerate(
        zip(selected_scores, methods, mean_scores)
    ):
        label = "\n{0:.3f}".format(mean)
        val = gaussian_filter1d(value, sigma=3, mode="nearest")

        plt.plot(val, label=label, c=colors[z], ls=ls[z], lw=lw[z])

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(
        loc=8,
        bbox_to_anchor=(0.5, 0.97),
        fancybox=False,
        shadow=True,
        ncol=4,
        fontsize=7,
        frameon=False,
    )

    plt.grid(ls=":", c=(0.7, 0.7, 0.7))
    plt.xlim(0, 200)
    axx = plt.gca()
    axx.spines["right"].set_visible(False)
    axx.spines["top"].set_visible(False)

    # plt.title(
    #     "%s %s\n%s" % (clfs[j], dependency[k][:], metrics[i]),
    #     fontfamily="serif",
    #     y=1.25,
    #     fontsize=8,
    # )
    plt.ylim(0.0, 1.0)
    plt.xticks(fontfamily="serif")
    plt.yticks(fontfamily="serif")
    plt.ylabel(metrics[i], fontfamily="serif", fontsize=6)
    plt.xlabel("chunks", fontfamily="serif", fontsize=6)
    plt.tight_layout()
    plt.savefig("plots/experiment3/runs/%s/3_%s_%s_%s.eps" % (what, clfs[j], metrics[i], dependency[k]), bbox_inches='tight', dpi=250, pad_inches=0.0)
    plt.close()

def plot_radars(
    methods, metrics, table, classifier_name, parameter_name, what
):
    """
    Strach.
    """
    columns = ["group"] + methods
    df = pd.DataFrame(columns=columns)
    for i in range(len(table)):
        df.loc[i] = table[i]
    df = pd.DataFrame()
    df["group"] = methods
    for i in range(len(metrics)):
        df[table[i][0]] = table[i][1:]
    groups = list(df)[1:]
    N = len(groups)

    # nie ma nic wspolnego z plotem, zapisywanie do txt texa
    # print(df.to_latex(index=False), file=open("tables/%s.tex" % (filename), "w"))

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # No shitty border
    ax.spines["polar"].set_visible(False)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, metrics)

    # Adding plots
    for i in range(len(methods)):
        values = df.loc[i].drop("group").values.flatten().tolist()
        values += values[:1]
        values = [float(i) for i in values]
        ax.plot(
            angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
        )

    # Add legend
    plt.legend(
        loc="lower center",
        ncol=3,
        columnspacing=1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.32),
        fontsize=6,
    )

    # Add a grid
    plt.grid(ls=":", c=(0.7, 0.7, 0.7))

    # Add a title
    plt.title("%s %s" % (clfs[j], parameter_name), size=8, y=1.08, fontfamily="serif")
    plt.tight_layout()

    # Draw labels
    a = np.linspace(0, 1, 6)
    plt.yticks(a[1:], ["%.1f" % f for f in a[1:]], fontsize=6, rotation=90)
    plt.ylim(0.0, 1.0)
    plt.gcf().set_size_inches(4, 3.5)
    plt.gcf().canvas.draw()
    angles = np.rad2deg(angles)

    ax.set_rlabel_position((angles[0] + angles[1]) / 2)

    har = [(a >= 90) * (a <= 270) for a in angles]

    for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
        x, y = label.get_position()
        print(label, angle)
        lab = ax.text(
            x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
        )
        lab.set_rotation(angle)

        if har[z]:
            lab.set_rotation(180 - angle)
        else:
            lab.set_rotation(-angle)
        lab.set_verticalalignment("center")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
        x, y = label.get_position()
        print(label, angle)
        lab = ax.text(
            x,
            y,
            label.get_text(),
            transform=label.get_transform(),
            fontsize=4,
            c=(0.7, 0.7, 0.7),
        )
        lab.set_rotation(-(angles[0] + angles[1]) / 2)

        lab.set_verticalalignment("bottom")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig("plots/experiment3/radars/%s/3_%s_%s.eps" % (what, classifier_name, parameter_name), bbox_inches='tight', dpi=250, pad_inches=0.0)
    plt.close()

for j, clf in enumerate(clfs):
    print("\n###\n### %s\n###\n" % (clf))
    for i, metric in enumerate(metrics):
        print("\n---\n--- %s\n---\n" % (metric))
        # KLASYFIKATOR, CHUJ, DIST, LABELNOISE, METHOD, CHUNK, METRYKA
        sub_scores = scores[j, :, :, :, :, :, i]
        # print(sub_scores.shape)

        # LABNO
        # CHUJ, DIST, LABELNOISE, METHOD, CHUNK
        reduced_scores = np.mean(sub_scores, axis=0)
        reduced_scores = np.mean(reduced_scores, axis=0)
        table = []
        header = ["LN"] + methods
        for k, label_noise in enumerate(ln):
            # LABELNOISE, METHOD, CHUNK
            selected_scores = reduced_scores[k]
            mean_scores = np.mean(selected_scores, axis=1)
            table.append([label_noise] + ["%.3f" % score for score in mean_scores])

            plot_runs(clfs, metrics, selected_scores, methods, mean_scores, ln, "label_noise")

        print(tabulate(table, headers=header))
        print("")

        # DISTRIBUTION
        # CHUJ, DIST, LABELNOISE, METHOD, CHUNK
        reduced_scores = np.mean(sub_scores, axis=0)
        reduced_scores = np.mean(reduced_scores, axis=1)
        table = []
        header = ["Dist"] + methods
        for k, distribution in enumerate(dist):
            # LABELNOISE, METHOD, CHUNK
            selected_scores = reduced_scores[k]
            mean_scores = np.mean(selected_scores, axis=1)
            table.append([distribution] + ["%.3f" % score for score in mean_scores])

            plot_runs(clfs, metrics, selected_scores, methods, mean_scores, dist, "distributions")

        # print(table)
        print(tabulate(table, headers=header))
        print("")

        # Drift
        # CHUJ, DIST, LABELNOISE, METHOD, CHUNK
        reduced_scores = np.mean(sub_scores, axis=1)
        reduced_scores = np.mean(reduced_scores, axis=1)
        table = []
        header = ["Drift"] + methods
        for k, drift in enumerate(drifts):
            # LABELNOISE, METHOD, CHUNK
            selected_scores = reduced_scores[k]
            mean_scores = np.mean(selected_scores, axis=1)
            table.append([drift] + ["%.3f" % score for score in mean_scores])

            plot_runs(clfs, metrics, selected_scores, methods, mean_scores, drifts, "drift_type")

            # Drift evaluation
            # for i in range(selected_scores.shape[0]):
            #     from csm import DriftEvaluator
            #     eval_ready_scores = selected_scores[i].reshape(1,199,1)
            #     drift_evaluator = DriftEvaluator(scores=eval_ready_scores, drift_indices=[100])
            #
            #     max_performance_loss = drift_evaluator.get_max_performance_loss()
            #     recovery_lengths = drift_evaluator.get_recovery_lengths()
            #     print(methods[i])
            #     print(max_performance_loss)
            #     print(recovery_lengths)

        # print(table)
        print(tabulate(table, headers=header))
        print("")

# RADAR DIAGRAMS

for j, clf in enumerate(clfs):
    print("\n###\n### %s\n###\n" % (clf))
    for i, drift in enumerate(drifts):
        print("\n---\n--- %s\n---\n" % (drift))
        # KLASYFIKATOR, CHUJ, DIST, LABELNOISE, METHOD, CHUNK, METRYKA
        sub_scores = scores[j, i, :, :, :, :, :]

        # Metryka
        # DIST, LABELNOISE, METHOD, CHUNK, METRYKA
        reduced_scores = np.mean(sub_scores, axis=0)
        reduced_scores = np.mean(reduced_scores, axis=0)
        table = []
        header = ["Metric"] + methods
        for k, metric in enumerate(metrics):
            # METHOD, CHUNK, Metryka
            selected_scores = reduced_scores[:,:,k]
            mean_scores = np.mean(selected_scores, axis=1)
            table.append([metric] + ["%.3f" % score for score in mean_scores])

        # print(table)
        print(tabulate(table, headers=header))
        print("")

        plot_radars(methods, metrics, table, clf, drift, "drift_type")

    for i, distribution in enumerate(dist):
        print("\n---\n--- %s\n---\n" % (distribution))
        # KLASYFIKATOR, CHUJ, DIST, LABELNOISE, METHOD, CHUNK, METRYKA
        sub_scores = scores[j, :, i, :, :, :, :]

        # Metryka
        # CHUJ, LABELNOISE, METHOD, CHUNK, METRYKA
        reduced_scores = np.mean(sub_scores, axis=0)
        reduced_scores = np.mean(reduced_scores, axis=0)
        table = []
        header = ["Metric"] + methods
        for k, metric in enumerate(metrics):
            # METHOD, CHUNK, Metryka
            selected_scores = reduced_scores[:,:,k]
            mean_scores = np.mean(selected_scores, axis=1)
            table.append([metric] + ["%.3f" % score for score in mean_scores])

        # print(table)
        print(tabulate(table, headers=header))
        print("")

        plot_radars(methods, metrics, table, clf, distribution, "distributions")

    for i, label_noise in enumerate(ln):
        print("\n---\n--- %s\n---\n" % (label_noise))
        # KLASYFIKATOR, CHUJ, DIST, LABELNOISE, METHOD, CHUNK, METRYKA
        sub_scores = scores[j, :, :, i, :, :, :]

        # Metryka
        # CHUJ, LABELNOISE, METHOD, CHUNK, METRYKA
        reduced_scores = np.mean(sub_scores, axis=0)
        reduced_scores = np.mean(reduced_scores, axis=0)
        table = []
        header = ["Metric"] + methods
        for k, metric in enumerate(metrics):
            # METHOD, CHUNK, Metryka
            selected_scores = reduced_scores[:,:,k]
            mean_scores = np.mean(selected_scores, axis=1)
            table.append([metric] + ["%.3f" % score for score in mean_scores])

        # print(table)
        print(tabulate(table, headers=header))
        print("")

        plot_radars(methods, metrics, table, clf, label_noise, "label_noise")
