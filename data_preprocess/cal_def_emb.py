import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-roberta-large-v1')

def main():
    
    with open('../data/words2defs.json') as f:
        word2def = json.loads(f.read())
    
    def2emb = {}
    for definitions in tqdm(word2def.values()):
        for sense in definitions:
            def2emb[sense] = model.encode(sense,
                                          convert_to_tensor=True)
            
    with open('../data/definition_emb.pickle', 'wb') as f:
        pickle.dump(def2emb, f)
    
if __name__ == '__main__':
    main()