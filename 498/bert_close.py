import pickle
import numpy as np
from utils import Config, safe_pickle_dump, strip_version
from sklearn.neighbors import NearestNeighbors
db = pickle.load(open(Config.db_path, 'rb'))
# X = pickle.load(open('elmo_embed.p', 'rb'))
db = pickle.load(open('bert_fine_tune.p', 'rb'))
X = np.array(list(db.values()))
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
# print(len(elmo_rv))
# print(elmo_rv[0].shape)
# X = X / np.linalg.norm(X, axis=0)
# print( np.linalg.norm(X[0]))
pids = list(db.keys())
# B = N
# ds = -np.asarray(np.dot(X, X.T)) #NxD * DxB => NxB
# IX = np.argsort(ds, axis=0) # NxB
# pid = '1407.2515'
# ID = pids.index(pid)
# distances, indices = nbrs.kneighbors([X[ID]])
# dist = X - X[ID]
# ecl = np.linalg.norm(dist, axis=1)
# print(len(ecl))
# IX = np.argsort(ecl) 
# print(IX.shape)
ARXIV_PATH = 'https://arxiv.org/abs/'
# print(ARXIV_PATH + pids[ID])
def get_similar(pid):
    pid = strip_version(pid)
    ID = pids.index(pid)
    distances, indices = nbrs.kneighbors([X[ID]])
    return [ pids[i] for i in indices[0]] 
# pid = '1904.07460'
# print(get_similar(pid))

# for i in range(10):
#     print(ARXIV_PATH+pids[indices[0][i]], distances[0][i])
#     # print(IX[ID][i])
#     # print(IX[i][ID])
    # print(db[pids[IX[i][ID]]]['id'])
    # print(ARXIV_PATH + pids[IX[i]])

