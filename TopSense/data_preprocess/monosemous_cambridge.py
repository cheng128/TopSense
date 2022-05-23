import json
import spacy
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append('..')
from util import tokenize_processor


def load_data():
    
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]

    return cambridge
        
def build_data(cambridge):
    
    cambridge_examples = []
    for line in cambridge:
        examples = [sent['en'] for sent in line['examples']]
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
    
    monosemous = {}
    
    for headword, pos_data in word2data.items():
        for pos, definitions in pos_data.items():
            if len(definitions) == 1:
                monosemous[headword] = {'pos': pos,
                                        'examples': []}
                
    return monosemous

def convert_pos(token_pos):
    if token_pos in ['PROPN', 'PRON', 'NOUN']:
        pos = 'noun'
    else:
        pos = token_pos.lower()
        
    return pos

def clear_data(monosemous_data, word2data):
    delete_list = []
    for key, value in monosemous_data.items():
        if not value['examples']:
            delete_list.append(key)
            continue
        pos = value['pos']
        data = word2data[key][pos]
        word_id = data[0]['id']
        monosemous_data[key]['id'] = word_id
    
    for key in delete_list:
        if not monosemous_data[key]['examples']:
            del monosemous_data[key]
            
    return monosemous_data

def save_data(word2data, monosemous):
    with open('../data/cambridge_word2data.json', 'w') as f:
        f.write(json.dumps(word2data))
    
    with open('../data/monosemous_data.json', 'w') as f:
        f.write(json.dumps(monosemous))

def main():
    
    cambridge = load_data()
    cambridge_examples, word2data = build_data(cambridge)
    monosemous = find_monosemous_word(word2data)
    
    for example in tqdm(cambridge_examples):
        sent_tokens = tokenize_processor(example)
        for token in sent_tokens:
            headword = token['lemma']
            token_pos = convert_pos(token['pos'])
            if headword in monosemous and monosemous[headword]['pos'] == token_pos:
                monosemous[headword]['examples'].append(example)

    clear_data(monosemous, word2data)
    save_data(word2data, monosemous)
    
    return

if __name__ == '__main__':
    main()