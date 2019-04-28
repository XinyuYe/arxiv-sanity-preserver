import os
import pickle
from utils import Config, safe_pickle_dump
from gensim.models.doc2vec import Doc2Vec, TaggedDocument 
# read database
db = pickle.load(open(Config.db_path, 'rb'))

# read all text files for all papers into memory
txt_paths, pids = [], []
n = 0
for pid,j in db.items():
  n += 1
  idvv = '%sv%d' % (j['_rawid'], j['_version'])
  txt_path = os.path.join('data', 'txt', idvv) + '.pdf.txt'
  if os.path.isfile(txt_path): # some pdfs dont translate to txt
    with open(txt_path, 'r') as f:
      txt = f.read()
    if len(txt) > 1000 and len(txt) < 500000: # 500K is VERY conservative upper bound
      txt_paths.append(txt_path) # todo later: maybe filter or something some of them
      pids.append(idvv)
      print("read %d/%d (%s) with %d chars" % (n, len(db), idvv, len(txt)))
    else:
      print("skipped %d/%d (%s) with %d chars: suspicious!" % (n, len(db), idvv, len(txt)))
  else:
    print("could not find %s in txt folder." % (txt_path, ))
print("in total read in %d text files out of %d db entries." % (len(txt_paths), len(db)))


print("precomputing nearest neighbor queries in batches...")
# X = X.todense() # originally it's a sparse matrix
sim_dict = {}
# batch_size = 200
# for i in range(0,len(pids),batch_size):
#   i1 = min(len(pids), i+batch_size)
#   xquery = X[i:i1] # BxD
#   ds = -np.asarray(np.dot(X, xquery.T)) #NxD * DxB => NxB
#   IX = np.argsort(ds, axis=0) # NxB
#   for j in range(i1-i):
#     sim_dict[pids[i+j]] = [pids[q] for q in list(IX[:50,j])]
#   print('%d/%d...' % (i, len(pids)))

model = Doc2Vec.load("d2v.model")
for pid in pids:
  tmp = []
  try:
    tmp = model.docvecs.most_similar(pid)
  except:
    tmp = []
  sim_dict[pid] = [sim_pid for sim_pid, distance in tmp]

print("writing", Config.sim_path)
safe_pickle_dump(sim_dict, Config.sim_path)
