import os
import sys
import time
import shutil
import pickle

from utils import Config, strip_version
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# # get model
# have = set(os.listdir(Config.txt_dir))
# files = os.listdir(Config.pdf_dir)
# tagged_data = []
# # there was a ,start=1 here that I removed, can't remember why it would be there. shouldn't be, i think.
# for i, f in enumerate(files):
#     txt_basename = f + '.txt'
#     # if txt_basename in have:
#     #     print('%d/%d skipping %s, already exists.' %
#     #           (i, len(files), txt_basename, ))
#     #     continue

#     # pdf_path = os.path.join(Config.pdf_dir, f)
#     txt_path = os.path.join(Config.txt_dir, txt_basename)
#     doc_id = f.split('.pdf')[0]
#     with open(txt_path) as txt_file:
#         tagged_data += TaggedDocument(
#             words=gensim.utils.simple_preprocess(txt_file.read()), tags=[doc_id]),

# max_epochs = 5
# vec_size = 20
# alpha = 0.025

# model = Doc2Vec(size=vec_size,
#                 alpha=alpha,
#                 min_alpha=0.00025,
#                 min_count=2,
#                 dm=0)

# model.build_vocab(tagged_data)

# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
#     model.train(tagged_data,
#                 total_examples=model.corpus_count,
#                 epochs=model.iter)
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha

# model.save("d2v.model")
# print("Model Saved")

# load generated model
model = Doc2Vec.load("d2v.model")
# tokens = 'Node Attribution Method'.lower().split()
# tokens = 'execute many different tasks depending on given task descriptions'.lower().split()
# tokens = 'novel non-parametric gradientbased policy, graph reward propagation, to pre-train'.split()
# tokens = 'subtask graph execution'
# tokens = tokens.lower().strip().split() # split by spaces
# new_vector = model.infer_vector(tokens)
# to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1903.08616v1')
query_doc = '1904.07460v1'
similar_doc = model.docvecs.most_similar(query_doc)
# similar_doc = model.docvecs.most_similar('1904.04994v1')
# similar_doc = model.docvecs.most_similar([new_vector])
db = pickle.load(open(Config.db_path, 'rb'))
ARXIV_PATH = 'https://arxiv.org/abs/'
def get_tags(doc_id):
    return [tag['term'] for tag in db[doc_id]['tags']]
# query_tags =set(get_tags(strip_version(query_doc)))
# print(query_tags)

# for doc in similar_doc:
#     # print(db[doc[0].split('v')[0]]['tags']['term'])
#     doc_tags = get_tags(strip_version(query_doc))
#     inter = set(doc_tags) & query_tags
#     print(inter)
    # print(get_tags(doc[0].split('v')[0]))

    # print(ARXIV_PATH+doc[0], doc[1])


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])
top_n = 10
def get_similar(doc_id):
    return [ doc for doc, _ in model.docvecs.most_similar(doc_id)[:top_n]]
# print(get_similar('1809.06647v3'))

# files = set(os.listdir(Config.txt_dir))
# total_sim = 0
# file_num = 0
# for i, f in enumerate(files):
#     doc_id = f.split('.pdf')[0]
#     # print(doc_id)
#     query_tags =set(get_tags(strip_version(doc_id)))
#     similar_doc = get_similar(doc_id)
#     file_num += 1
#     for doc in similar_doc:
#         # print(db[doc[0].split('v')[0]]['tags']['term'])
#         doc_tags = get_tags(strip_version(doc))
#         inter = set(doc_tags) & query_tags
#         if inter: 
#             total_sim += 1
#         # else:
#         #     print(doc_tags, query_tags)

# print(f'{total_sim}/{file_num*top_n}: {total_sim/file_num/top_n}')