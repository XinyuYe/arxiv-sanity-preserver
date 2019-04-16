import pickle
import numpy as np
db = pickle.load(open('bert_out_big.p', 'rb'))
# db = pickle.load(open('bert_out.p', 'rb'))
print(len(db))
X = np.array(list(db.values()))
# normalization
X = X / np.linalg.norm(X, axis=0)
pids = list(db.keys())
# B = N
ds = -np.asarray(np.dot(X, X.T)) #NxD * DxB => NxB
IX = np.argsort(ds, axis=0) # NxB
pid = '1807.07665'
ID = pids.index(pid)
# print(IX.shape)
ARXIV_PATH = 'https://arxiv.org/abs/'
print(ARXIV_PATH + pids[ID])
for i in range(10):
    # print(IX[ID][i])
    print(IX[i][ID])
    print(ARXIV_PATH + pids[IX[i][ID]])
