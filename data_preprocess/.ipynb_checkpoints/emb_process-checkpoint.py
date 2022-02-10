import json
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-roberta-large-v1')

with open('../data/jsonl_file/refactor_True_0.0_top3_map.jsonl') as f:
    data = [json.loads(line) for line in f.readlines()]
    
sent_dict = defaultdict(list)
for i in data:
    sent_dict[i['category'].lower()].append(i['en_def'])

for category in sent_dict:
    sent_dict[category].append(category)
    
emb_dict = {}
for topic, sents in sent_dict.items():
    emb_dict[topic] = model.encode(sents, convert_to_tensor=True)
    
with open('../data/topic_embs.pickle', 'wb') as f:
    pickle.dump(emb_dict, f)