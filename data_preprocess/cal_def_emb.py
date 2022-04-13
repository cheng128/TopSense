import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-t5-xl')

def main():
    
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]
    
    def2emb = {}
    for line in tqdm(cambridge):
        if line['pos'] == 'noun':
            sense = line['en_def']
#             examples = [sent['en'] for sent in line['examples']]
#             print(examples)
#             sentences = [sense].extend(examples)
            sentences = [sense]
            def2emb[sense] = model.encode(sentences,
                                          convert_to_tensor=True)
        
    with open('../data/t5-xl_def_emb.pickle', 'wb') as f:
        pickle.dump(def2emb, f)
    
if __name__ == '__main__':
    main()