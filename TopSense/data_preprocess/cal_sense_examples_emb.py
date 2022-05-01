import json
import pickle
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-t5-xl')

def main():

    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]
    
    def2emb = defaultdict(dict)
    for line in tqdm(cambridge):
        word = line['headword']
        pos = line['pos']
        sense = line['en_def']   
        examples = [sent['en'] for sent in line['examples']]
        def2emb[(word, pos)][sense] = {'sense_emb' :model.encode([sense],
                                                    convert_to_tensor=True),
                                        'examples_embs': ''}
        if examples:
            def2emb[(word, pos)][sense]['examples_embs'] = model.encode(examples, 
                                                            convert_to_tensor=True)

    
    with open(f'../data/sentence-t5-xl_sense_examples_embs.pickle', 'wb') as f:
        pickle.dump(def2emb, f)
    
if __name__ == '__main__':
    main()