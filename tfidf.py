import os
import json
import time
import pickle
import argparse
import dateutil.parser
from random import shuffle, randrange, uniform

import numpy as np
from sqlite3 import dbapi2 as sqlite3
from hashlib import md5
from flask import Flask, request, session, url_for, redirect, \
    render_template, abort, g, flash, _app_ctx_stack

from utils import safe_pickle_dump, strip_version, isvalidid, Config


def connect_db():
    sqlite_db = sqlite3.connect(Config.database_path)
    sqlite_db.row_factory = sqlite3.Row  # to return dicts rather than tuples
    return sqlite_db


def papers_search(qraw):
    qparts = qraw.lower().strip().split()  # split by spaces
    # use reverse index and accumulate scores
    scores = []
    for pid, p in db.items():
        score = sum(SEARCH_DICT[pid].get(q, 0) for q in qparts)
        if score == 0:
            continue  # no match whatsoever, dont include
        # give a small boost to more recent papers
        score += 0.0001*p['tscore']
        scores.append((score, p))
    scores.sort(reverse=True, key=lambda x: x[0])  # descending
    out = [x[1] for x in scores if x[0] > 0]
    return out


db = pickle.load(open(Config.db_serve_path, 'rb'))


def search(search_term):
    # DATE_SORTED_PIDS = cache['date_sorted_pids']
    # TOP_SORTED_PIDS = cache['top_sorted_pids']
    cache = pickle.load(open(Config.serve_cache_path, "rb"))
    SEARCH_DICT = cache['search_dict']
    # search_term = 'Node Attribution Method'
    result = [p['id'] for p in papers_search(search_term)[:10]]
    print('\n'.join(result))


def papers_similar(pid):
    sim_dict = pickle.load(open(Config.sim_path, "rb"))
    rawpid = strip_version(pid)

    # check if we have this paper at all, otherwise return empty list
    if not rawpid in db:
        return []

    # check if we have distances to this specific version of paper id (includes version)
    if pid in sim_dict:
        # good, simplest case: lets return the papers
        return [db[strip_version(k)] for k in sim_dict[pid]]
    else:
        # ok we don't have this specific version. could be a stale URL that points to,
        # e.g. v1 of a paper, but due to an updated version of it we only have v2 on file
        # now. We want to use v2 in that case.
        # lets try to retrieve the most recent version of this paper we do have
        kok = [k for k in sim_dict if rawpid in k]
        if kok:
            # ok we have at least one different version of this paper, lets use it instead
            id_use_instead = kok[0]
            return [db[strip_version(k)] for k in sim_dict[id_use_instead]]
        else:
            # return just the paper. we dont have similarities for it for some reason
            return [db[rawpid]]

# search('Node Attribution Method')
# pid = '1904.07460'
# rv = papers_similar(pid)
# rv = [p['id'] for p in rv[:10] ]
# print('\n'.join(rv))
sim_dict = pickle.load(open(Config.sim_path, "rb"))
def get_similar(pid):
  rawpid = strip_version(pid)
  if pid in sim_dict:
      # good, simplest case: lets return the papers
      return [k for k in sim_dict[pid]]
  return []
  # return papers_similar(doc_id)

# print(get_similar('1809.06647v3'))
# print(get_similar('1904.07460'))

