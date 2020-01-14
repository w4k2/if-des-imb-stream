"""
Analiza zależności od szumu
"""
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
from math import pi

methods = ["OOB", "UOB", "SEA", "SSEA"]
label_noises = [
    "0.01"
]
ln = [a.replace('.','-') for a in label_noises]
distributions = ["0.05", "0.10"]
dist = [a.replace('.','-') for a in distributions]
drifts = ["gradual", "sudden"]
metrics = ["bac", "gmean", "f1", "precision", "recall", "specificity"]
clfs = ["GNB"]

scores = np.load("scores.npy")

# print(scores.shape)

def plot_runs(clfs, metrics, selected_scores, methods, mean_scores, dependency, what):
    fig = plt.figure()
    ax = plt.axes()
    for value, label, mean in zip(selected_scores, methods, mean_scores):
        label += '\n{0:.3f}'.format(mean)
        # plt.plot(medfilt(value, 3), label=label);
        plt.plot(value, label=label);
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=False, shadow=True, ncol=5, frameon=False)
    plt.title("%s %s %s" % (clfs[j], metrics[i], dependency[k]))
    plt.ylim(0.0,1.0)
    plt.savefig("plots/experiment1/runs/%s/%s_%s_%s" % (what, clfs[j], metrics[i], dependency[k]), bbox_inches='tight', dpi=250)
    plt.close()

def plot_radars(methods, metrics, table, classifier_name, parameter_name, what):
    columns = ["group"] + methods
    df = pd.DataFrame(columns=columns)
    for i in range(len(table)):
        df.loc[i] = table[i]

    print(df)

    df = pd.DataFrame()
    df['group'] = methods
    for i in range(len(metrics)):
        df[table[i][0]] = table[i][1:]

    groups = list(df)[1:]
    N = len(groups)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], groups)

    for label in ax.get_xticklabels():
        label.set_rotation(120)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.15,0.20,0.25,0.30,0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1.00],
        ["0.15","0.20","0.25","0.30","0.35", "0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"],
        fontsize=6,
    )
    plt.ylim(0.15, 1.0)

    # print(df)
    # exit()
    # Adding plots
    for i in range(len(methods)):
        values = df.loc[i].drop('group').values.flatten().tolist()
        values += values[:1]
        values = [float(i) for i in values]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.iloc[i,0])


    # Add legend
    plt.legend(loc="lower center", ncol=5, columnspacing=1, frameon=False, bbox_to_anchor=(0.5, -0.2))
    # Add a title
    plt.title("%s %s" % (classifier_name, parameter_name), size=11, y=1.08)
    plt.savefig("plots/experiment1/radars/%s/%s_%s.png" % (what, classifier_name, parameter_name), bbox_inches='tight', dpi=250)
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
        for k, label_noise in enumerate(label_noises):
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
        for k, distribution in enumerate(distributions):
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

    for i, distribution in enumerate(distributions):
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

    for i, label_noise in enumerate(label_noises):
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
