import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# dataset = 'datasets/covtypeNorm-1-2vsAll' # 171 - 437, 1000 (266) IR: 4
dataset = 'datasets/covtypeNorm-1-2vsAll-pruned'
# dataset = 'datasets/poker-lsn-1-2vsAll' # 0 - 360, 1000 (360) IR: 10
dataset = 'datasets/poker-lsn-1-2vsAll-pruned'
# dataset = 'datasets/shuttle-5vsAll' # 0-232, 250 (232) IR: 17
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",", dtype="|U5")
X = dataset[:, :-1]
y = dataset[:, -1]

le = LabelEncoder()
y = le.fit_transform(y)

d_classes, d_counts = np.unique(y, return_counts=True)
print(d_counts)
# exit()
chunk_size = 1000
n_chunks = round(X.shape[0]/chunk_size)

print(n_chunks)

maj = 0
min = 0

list_ok = []
list_no = []

consecutive_ok = 0
consecutive_no = 0

for chunk in range(n_chunks):

    ind_a = chunk_size*chunk
    ind_b = chunk_size*(chunk+1)
    chunk_y = y[ind_a:ind_b]
    classes, counts = np.unique(chunk_y, return_counts=True)
    maj += counts[0]
    if counts.shape[0] == 2:
        min += counts[1]
    if classes.shape[0] == 2 and counts[1] >= 5:
        # print("Chunk: %i OK" % chunk)
        consecutive_ok += 1
        if consecutive_no != 0:
            list_no.append(consecutive_no)
        consecutive_no = 0
    else:
        # print("Chunk: %i NO" % chunk)
        consecutive_no += 1
        if consecutive_ok != 0:
            list_ok.append(consecutive_ok)
        consecutive_ok = 0

if consecutive_no != 0:
    list_no.append(consecutive_no)
if consecutive_ok != 0:
    list_ok.append(consecutive_ok)

print("IR: %.0f" % (maj/min))
print(list_ok)
print(list_no)

print(sum(list_no) + sum(list_ok))
