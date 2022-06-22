import json
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append('..')
from util import tokenize_processor

SPACY_MAP = {'ADJ': 'adjective', 'ADV': 'adverb', 'PROPN': 'noun',
             'PRON': 'noun', 'NOUN': 'noun', 'VERB': 'verb'}

def load_data():
    
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]

    return cambridge
        
def build_data(cambridge):
    
    cambridge_examples = []
    for line in cambridge:
        examples = [[line['headword'],sent['en']] 
                    for sent in line['examples']]
        cambridge_examples.extend(examples)
    
    word2data = defaultdict(dict)
    
    for line in cambridge:
        headword = line['headword']
        pos = line['pos']
        if pos in word2data[headword]:
            word2data[headword][pos].append(line)
        else:
            word2data[headword][pos] = [line]
    
    return cambridge_examples, word2data

def find_monosemous_word(word2data):
    
    monosemous = defaultdict(dict)
    
    for headword, pos_data in word2data.items():
        for pos, definitions in pos_data.items():
            if len(definitions) == 1:
                word_id = definitions[0]['id']
                monosemous[headword][pos] = {'id': word_id,
                                            'examples':[]}
    return monosemous

def clear_data(monosemous_data, word2data):
    delete_list = []
    for key, data in monosemous_data.items():
        for pos, pos_data in data.items():
            if not pos_data['examples']:
                delete_list.append([key, pos])
                continue
            else:
                monosemous_data[key][pos]['examples'] = list(set(pos_data['examples']))

    for key, pos in delete_list:
        if len(monosemous_data[key]) == 1:
            del monosemous_data[key]
        else:
            del monosemous_data[key][pos]
            
    return monosemous_data

def save_data(monosemous, monosemous_filename, word2data, word2data_filename):
    
    with open(monosemous_filename, 'w') as f:
        f.write(json.dumps(monosemous))
    
    with open(word2data_filename, 'w') as f:
        f.write(json.dumps(word2data))

def main():
    
    cambridge = load_data()
    cambridge_examples, word2data = build_data(cambridge)
    monosemous = find_monosemous_word(word2data)
    
    for headword, example in tqdm(cambridge_examples):
        sent_tokens = tokenize_processor(example)
        for token in sent_tokens:
            target = token['lemma']
            # is not one of the origin examples
            if headword.lower() != target.lower():
                token_pos = SPACY_MAP.get(token['pos'], '')
                if token_pos:
                    if target in monosemous and token_pos in monosemous[target]:
                        monosemous[target][token_pos]['examples'].append(example)

    clear_data(monosemous, word2data)
    save_data(monosemous, '../data/monosemous_data.json', word2data, '../data/cambridge_word2data.json')
    
    return

if __name__ == '__main__':
    main()