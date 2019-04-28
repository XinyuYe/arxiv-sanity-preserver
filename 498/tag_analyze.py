import os
import sys
import time
import shutil
import pickle

from utils import Config, strip_version
import gensim
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from doc2vec import get_similar
# from tfidf import get_similar
# from elmo_close import get_similar
# from elmo_nn import get_similar
from rand_choose import get_similar
# from bert_nn import get_similar
# from bert_close import get_similar

db = pickle.load(open(Config.db_path, 'rb'))
def get_tags(doc_id):
    return [tag['term'] for tag in db[doc_id]['tags']]

top_n = 10
# def get_similar(doc_id):
#     return get_similar_t(doc_id)
    # return d2v_get_similar(doc_id)
    # return [ doc for doc, _ in model.docvecs.most_similar(doc_id)[:top_n]]

files = set(os.listdir(Config.txt_dir))
total_sim = 0
file_num = 0
for i, f in enumerate(files):
    doc_id = f.split('.pdf')[0]
    # print(doc_id)
    query_tags =set(get_tags(strip_version(doc_id)))
    similar_doc = get_similar(doc_id)
    # if not similar_doc: continue
    file_num += 1
    for doc in similar_doc:
        # print(db[doc[0].split('v')[0]]['tags']['term'])
        doc_tags = get_tags(strip_version(doc))
        inter = set(doc_tags) & query_tags
        if inter: 
            total_sim += 1
        # else:
        #     print(doc_tags, query_tags)

print(f'{total_sim}/{file_num*top_n}: {total_sim/file_num/top_n}')