import json
import pickle
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

def main():
    model_name = 'sentence-t5-xl'
    print(model_name)
    model = SentenceTransformer(model_name)

    with open('../data/jsonl_file/refactor_True_0_top3_map_cross_reference.jsonl') as f:
        data = [json.loads(line) for line in f.readlines()]


    sent_dict = defaultdict(list)
    for i in data:
        sent_dict[i['category'].lower()].append(i['en_def'])

    for category in sent_dict:
        sent_dict[category].append(category)


    emb_dict = {}
    for defs, sents in tqdm(sent_dict.items()):
        emb_dict[defs] = model.encode(sents, convert_to_tensor=True)


    with open(f'../data/topic_emb/{model_name}_topic_embs.pickle', 'wb') as f:
        pickle.dump(emb_dict, f)
    
if __name__ == '__main__':
    main()