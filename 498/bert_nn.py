import pickle
import numpy as np
from utils import Config, safe_pickle_dump, strip_version

db = pickle.load(open('bert_fine_tune.p', 'rb'))
# db = pickle.load(open('bert_out.p', 'rb'))
print(len(db))
X = np.array(list(db.values()))
# normalization
X = X / np.linalg.norm(X, axis=1, keepdims=1)
pids = list(db.keys())
# B = N
ds = -np.asarray(np.dot(X, X.T)) #NxD * DxB => NxB
# print(ds[0][0])
IX = np.argsort(ds, axis=0) # NxB
# pid = '1407.2515'
# pid = '1904.05856'
# pid = '1904.07460'
# ID = pids.index(pid)
# print(IX.shape)
# ARXIV_PATH = 'https://arxiv.org/abs/'
# print(ARXIV_PATH + pids[ID])
# for i in range(10):
#     # print(IX[ID][i])
#     # print(ds[ID][IX[i]])
#     # print(1+ds[ID][IX[i][ID]])
#     print(ARXIV_PATH + pids[IX[i][ID]])

def get_similar(pid):
    pid = strip_version(pid)
    ID = pids.index(pid)
    return [pids[IX[i][ID]] for i in range(1, 11)]
