import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-t5-xl')

def main():
    
#     with open('../data/words2defs.json') as f:
#         word2def = json.loads(f.read())
    
#     def2emb = {}
#     for definitions in tqdm(word2def.values()):
#         for sense in definitions:
#             def2emb[sense] = model.encode(sense,
#                                           convert_to_tensor=True)
            
#     with open('../data/definition_emb.pickle', 'wb') as f:
#         pickle.dump(def2emb, f)

    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]
    
    def2emb = {}
    for line in tqdm(cambridge):
        if line['pos'] == 'noun':
            sentences = [line['en_def']]
            def2emb[line['en_def']] = model.encode(sentences,
                                              convert_to_tensor=True)
        
    with open('../data/t5-xl_def_examples_emb.pickle', 'wb') as f:
        pickle.dump(def2emb, f)
    
if __name__ == '__main__':
    main()