import os
import sys
import time
import shutil
import pickle

from utils import Config
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# get model
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
#                 min_count=1,
#                 dm=1)

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

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1903.08616v1')
similar_doc = model.docvecs.most_similar('1807.07665v3')
ARXIV_PATH = 'https://arxiv.org/abs/'
for doc in similar_doc[:10]:
    print(ARXIV_PATH+doc[0])


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])
