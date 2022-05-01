import json
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

def load_data():
    with open('../data/wiki/all_simple_wiki_word_id2sents_highest.json') as f:
        word_id2sents = json.loads(f.read())
        
    return word_id2sents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=str)
    parser.add_argument('-n', type=int)
    
    args = parser.parse_args()
    version = args.v
    num = args.n
    
    word_id2sents = load_data()
    
    filename = f'../data/wiki/{num}_{version}_wiki_word_id2sents_highest.json'

    sample_word_id2sents = defaultdict(list)
    for word_id, value in tqdm(word_id2sents.items()):
        sentences = value[-1]
        if len(sentences) > num:
            sentences = random.sample(sentences, num)
        sample_word_id2sents[word_id].append(value[0])
        sample_word_id2sents[word_id].append(value[1])
        sample_word_id2sents[word_id].append(sentences)
    
    with open(filename, 'w') as f:
        f.write(json.dumps(sample_word_id2sents))
    
    return

if __name__ == '__main__':
    main()