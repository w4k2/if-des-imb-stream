import strlearn as sl
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

stream = sl.streams.ARFFParser("streams/example.arff", n_chunks=200, chunk_size=250)
eval = sl.evaluators.TestThenTrain(metrics=(sl.metrics.balanced_accuracy_score))
clf = sl.ensembles.SEA(base_estimator=DecisionTreeClassifier(random_state=42))

eval.process(stream, clf)

print(eval.scores)

plt.plot(eval.scores[0])
plt.savefig("test.png")
