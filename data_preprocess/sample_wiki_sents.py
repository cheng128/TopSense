import json
import random
import argparse
from tqdm import tqdm

def load_data():
    with open('../data/wiki/fix_t5-xl_wiki_word_id2sents.json') as f:
        word_id2sents = json.loads(f.read())
        
    return word_id2sents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    
    args = parser.parse_args()
    num = args.n
    
    word_id2sents = load_data()
    
    filename = f'../data/wiki/{num}-t5-xl_wiki_word_id2sents.json'

    for word_id, value in tqdm(word_id2sents.items()):
        sentences = value[-1]
        if len(sentences) > num:
            sentences = random.sample(sentences, num)
        value[-1] = sentences         
    
    with open(filename, 'w') as f:
        f.write(json.dumps(word_id2sents))
    
    return

if __name__ == '__main__':
    main()