import strlearn as sl
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
from csm import StratifiedBagging

stream = sl.streams.ARFFParser("Agrawal_i10_ln10_s.arff", chunk_size=250, n_chunks=200)
# stream = sl.streams.StreamGenerator(n_drifts=1, weights=[
#                                     0.8, 0.2], chunk_size=250, n_chunks=200, n_features=8, n_informative=8, n_redundant=0, n_clusters_per_class=1, random_state=1410, concept_sigmoid_spacing=999)
clf = sl.ensembles.SEA(base_estimator=GaussianNB(),
                       metric=sl.metrics.balanced_accuracy_score, n_estimators=5)
eval = sl.evaluators.TestThenTrain(
    metrics=(sl.metrics.balanced_accuracy_score))

eval.process(stream, clf)
stream.reset()

fig = plt.figure()
plt.plot(eval.scores[0])
plt.savefig("foo.png")
plt.close()
