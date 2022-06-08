import json
import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup
from monosemous_cambridge import *
import sys
sys.path.append('..')
from util import tokenize_processor

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def main():
    cambridge = load_data()
    cambridge_examples, word2data = build_data(cambridge)
    monosemous = find_monosemous_word(word2data)
    
    with open(f'../data/wiki/simple_en_wiki_link.txt') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            article = data['text']

            sentences = tokenizer.tokenize(article)
            
            for sent in sentences:
                sent_tokens = tokenize_processor(sent)
                for token in sent_tokens:
                    headword = token['lemma']
                    token_pos = convert_pos(token['pos'])
                    if headword in monosemous and token_pos in monosemous[headword]:
                        monosemous[headword][token_pos]['examples'].append(sent)
                        data = word2data[headword][token_pos]
                        assert(len(data) == 1)
                        word_id = data[0]['id']
                        monosemous[headword][token_pos]['id'] = word_id

    clear_data(monosemous, word2data)
    save_data(monosemous, '../data/wiki/wikipedia_monosemous.json',
              word2data, '../data/wiki/cambridge_word2data.json')
                
    
    
if __name__ == '__main__':
    main()