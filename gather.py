from strlearn.streams import StreamGenerator
import numpy as np

# Variables
clfs = ["GNB", "HT", "MLP"]
methods = ["OSEA", "OOB", "UOB", "MDET_BAC", "MDET_FSCORE"]
random_states = [1410]
distributions = [[0.95, 0.05], [0.9, 0.1]]
label_noises = [
    0.01,
    0.05
]
# css = [5, None]
# incremental = [False, True]

chuj = [(5, False), (None, False)]

n_chunks = 19
metrics = ["bac", "geometric_mean_score", "f_score", "precision", "recall", "specificity", "accuracy"]

scores = np.zeros(
    (
        len(clfs),
        len(random_states),
        len(chuj),
        # len(incremental),
        len(distributions),
        len(label_noises),
        # len(css),
        len(methods),
        n_chunks,
        len(metrics),
    )
)

print(scores.shape)

# Prepare streams
streams = {}
for i, clf in enumerate(clfs):
    for j, random_state in enumerate(random_states):
        for k, kurwa in enumerate(chuj):
            for l, distribution in enumerate(distributions):
                for m, flip_y in enumerate(label_noises):
                    # for n, spacing in enumerate(css):
                    spacing, drift_type = kurwa
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=random_state,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=1,
                        n_chunks=20,
                        chunk_size=250,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        results = np.load(
                            "results2/experiment1_%s/%s.npy" % (clf, stream)
                        )
                        scores[i, j, k, l, m] = results
scores = np.mean(scores, axis=1)
np.save("scores", scores)
print(scores, scores.shape)
