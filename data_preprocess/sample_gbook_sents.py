import json
import random
import argparse
from tqdm import tqdm

def load_data():
    with open('../data/gbook/all_sense_to_gbook_examples.json') as f:
        word_id2sents = json.loads(f.read())
        
    return word_id2sents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    
    args = parser.parse_args()
    num = args.n
    
    word_id2sents = load_data()
    
    filename = f'../data/gbook/{num}_sense_to_gbook_examples.json'

    for word_id, sentences in tqdm(word_id2sents.items()):
        if len(sentences) > num:
            sentences = random.sample(sentences, num)
        word_id2sents[word_id] = sentences
    
    with open(filename, 'w') as f:
        f.write(json.dumps(word_id2sents))
    
    return

if __name__ == '__main__':
    main()