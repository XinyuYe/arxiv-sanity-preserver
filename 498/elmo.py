from allennlp.commands.elmo import ElmoEmbedder
import pickle
from utils import Config, safe_pickle_dump
import gensim
elmo = ElmoEmbedder(
    options_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
    weight_file='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
)

db = pickle.load(open(Config.db_path, 'rb'))
summary_tokens = []
for pid, j in db.items():
    # idvv = '%sv%d' % (j['_rawid'], j['_version'])
    summary = j['summary'].replace('\n', ' ')
    summary = gensim.utils.simple_preprocess(summary)
    summary_tokens += summary,
print(len(summary_tokens))
elmo_embed = elmo.embed_batch(summary_tokens)
safe_pickle_dump(elmo_embed, 'elmo_embed.p')