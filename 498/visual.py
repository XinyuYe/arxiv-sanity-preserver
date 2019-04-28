#!/usr/bin/env python
# coding: utf-8

# In[92]:

import os
import sys
import numpy as np
import time
import shutil
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import Config
import gensim
from utils import Config, safe_pickle_dump
import urllib
import feedparser
import urllib.request as libreq
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# model= Doc2Vec.load("d2v.model")
# num_of_papers = len(model.docvecs) 
num_of_papers = 1100
# docvecs = []
# for i in range(num_of_papers):
#     docvecs.append(model.docvecs[i])
# docvecs = np.array(docvecs)


# In[93]:


category = {} 
db = pkl.load(open(Config.db_path, 'rb'))
for k,v in db.items():
    category[k] = v['tags'][0]['term']
set_of_category =  set(category.values())
print(set_of_category)
colors = []
root = int(np.ceil(np.power(len(set_of_category),1/4)))
for a in range(1,1+root):
    for b in range(1,1+root):
        for c in range(1,1+root):
            for d in range(1,1+root):
                colors.append((1/root*a,1/root*b,1/root*c,1/root*d))

corr = dict(zip(list(set_of_category),colors[:len(set_of_category)]))


# In[94]:


def TSE_visualization(X,query_dict,output_file=None):
    plt.figure(figsize=(16, 16)) 
    X_embedded = TSNE(n_components=2, n_iter=1000, perplexity=40, verbose=2, random_state=23).fit_transform(X)
    color = np.array([corr[category[query]] for k,query in query_dict.items()])
    # label = np.array([legend[category[query]] for k, query in query_dict.items()])
    plt.scatter(X_embedded[:,0],X_embedded[:,1],color=color)
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in corr.values()]
    plt.legend(markers, corr.keys(), numpoints=1, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.legend()
    if output_file:
        plt.savefig(output_file)
    plt.show()


# In[ ]:


# visualization of doc2vec
# doc2vec_dict = {i: model.docvecs.index_to_doctag(i)[:-2] for i in range(num_of_papers)}
# TSE_visualization(docvecs,doc2vec_dict,"doc2vec_visual.png")


# In[ ]:


# visualization of BERT
# db = pkl.load(open(Config.db_path, 'rb'))
db = pkl.load(open('bert_out_big.p', 'rb'))
# db = pkl.load(open('bert_fine_tune.p', 'rb'))
# db = pkl.load(open('elmo_embed.p', 'rb'))
# model = Doc2Vec.load("d2v.model")
X = np.array(list(db.values()))
# X = np.array(pkl.load(open('elmo_embed.p', 'rb')))
# X = X / np.linalg.norm(X, axis=0)
pids = list(db.keys())
# ds = -np.asarray(np.dot(X, X.T))
bert_dict = dict(zip(range(num_of_papers), pids))
TSE_visualization(X, bert_dict,"bert_visual.png")
