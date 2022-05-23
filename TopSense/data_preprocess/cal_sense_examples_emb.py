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
    word_pos2senses = defaultdict(dict)
    for line in tqdm(cambridge):
        word = line['headword']
        pos = line['pos']
        sense = line['en_def'] 
        if (word, pos) not in word_pos2senses:
            word_pos2senses[(word, pos)] = [sense]
        else:
            word_pos2senses[(word, pos)].append(sense)
        # examples = [sent['en'] for sent in line['examples']]
        # def2emb[(word, pos)][sense] = {'sense_emb' :model.encode([sense],
        #                                             convert_to_tensor=True).cpu(),
        #                                 'examples_embs': ''}
        # if examples:
        #     def2emb[(word, pos)][sense]['examples_embs'] = model.encode(examples, 
        #                                                     convert_to_tensor=True).cpu()

    for key, value in tqdm(word_pos2senses.items()):
        def2emb[key]['senses'] = value
        def2emb[key]['embs'] = model.encode(value, convert_to_tensor=True).cpu()
    
    with open(f'../data/sentence-t5-xl_sense_embs_cpu.pickle', 'wb') as f:
        pickle.dump(def2emb, f)
    
if __name__ == '__main__':
    main()