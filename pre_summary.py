"""
Preprocess the summary
"""
import os
import pickle
from random import shuffle, seed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import Config, safe_pickle_dump

import spacy

# read database
db = pickle.load(open(Config.db_path, 'rb'))

# spacy dict
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# read all summaries
txt_paths, pids = [], []
n = 0
for pid, j in db.items():
    n += 1
    idvv = '%sv%d' % (j['_rawid'], j['_version'])
    summary = j['summary'].replace('\n', ' ')
    summary_txt_path = os.path.join('data', 'summary', idvv) + '.txt'
    doc = nlp(summary)
    sents = doc.sents
    print(f'Wrting Summary: {idvv}')
    with open(summary_txt_path, 'w+') as f:
        # for sentence in sents:
        f.write(summary)
