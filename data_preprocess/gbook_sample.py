import json
import argparse


def load_data():
    with open('remap.word_id.topics.examples.json') as f:
        word_id2topics = json.loads(f.read())
        
    with open('../data/sense_to_gbook_examples.json') as f:
        gbook_data = json.loads(f.read())
    
    return word_id2topics, gbook_data
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str)
    args = parser.parse_args()
    
    num = args.n
    word_id2topics, gbook_data = load_data(num)
    
    
    
    
    sample_dict = {}
    for word_id, sents in gbook_data.items():
        if len(sents) <= num:
            sample_dict[word_id] == 
    
    return 

if __name__ == '__main__':
    main()