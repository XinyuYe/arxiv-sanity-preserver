import pickle
import numpy as np
# db = pickle.load(open('bert_fine_tune.p', 'rb'))
from utils import Config, safe_pickle_dump, strip_version
db = pickle.load(open(Config.db_path, 'rb'))
orig = pickle.load(open('elmo_embed.p', 'rb'))
# db = pickle.load(open('bert_out.p', 'rb'))
# print(len(db))
# X = np.array(list(db.values()))
# normalization
X = orig / np.linalg.norm(orig, axis=1, keepdims=1)
# print(X.shape)
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
ARXIV_PATH = 'https://arxiv.org/abs/'
# print(ARXIV_PATH + pids[ID])
# print(orig[ID])
# for i in range(0,6):
#     # print(IX[ID][i])
#     # print(orig[IX[i][ID]])
#     # print(1+ds[ID][IX[i][ID]], end=' ')
#     sim_pid = pids[IX[i][ID]]
#     print(ARXIV_PATH + sim_pid)
#     # print(db[sim_pid]['title'])
#     print('-----')

def get_similar(pid):
    pid = strip_version(pid)
    ID = pids.index(pid)
    return [pids[IX[i][ID]] for i in range(1, 11)]

# pid = '1904.07460' 
# print(get_similar(pid))