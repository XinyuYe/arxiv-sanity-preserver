import pickle
import numpy as np
# db = pickle.load(open('bert_fine_tune.p', 'rb'))
from utils import Config, safe_pickle_dump, strip_version
db = pickle.load(open(Config.db_path, 'rb'))

pids = list(db.keys())
def get_similar(pid):
    return np.random.choice(pids, 10)

# print(get_similar('1'))