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
    
    monosemous = defaultdict(dict)
    
    for headword, pos_data in word2data.items():
        for pos, definitions in pos_data.items():
            if len(definitions) == 1:
                monosemous[headword][pos] = {'id':'',
                                            'examples':[]}
                
    return monosemous

def convert_pos(token_pos):
    if token_pos in ['PROPN', 'PRON', 'NOUN']:
        pos = 'noun'
    else:
        pos = token_pos.lower()
        
    return pos

def clear_data(monosemous_data, word2data):
    delete_list = []
    for key, data in monosemous_data.items():
        for pos, pos_data in data.items():
            if not pos_data['examples']:
                delete_list.append([key, pos])
                continue

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
    
    for example in tqdm(cambridge_examples):
        sent_tokens = tokenize_processor(example)
        for token in sent_tokens:
            headword = token['lemma']
            token_pos = convert_pos(token['pos'])
            if headword in monosemous and token_pos in monosemous[headword]:
                monosemous[headword][token_pos]['examples'].append(example)
                data = word2data[headword][token_pos]
                assert(len(data) == 1)
                word_id = data[0]['id']
                monosemous[headword][token_pos]['id'] = word_id

    clear_data(monosemous, word2data)
    save_data(monosemous, '../data/monosemous_data.json', word2data, '../data/cambridge_word2data.json')
    
    return

if __name__ == '__main__':
    main()