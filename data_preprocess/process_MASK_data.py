""" This program can process brt to cambridge data
into training data.
"""
import json
import argparse
from tqdm import tqdm
from utils import handle_examples

def brt_groups_results(reserve=False):
    
    with open('../data/word_id.topics.examples.json') as f:
        data = json.loads(f.read())
    
    with open(f'../data/training_data/brt_masked_noun.tsv', 'a') as f:
        for v in tqdm(data.values()):
            if v['pos'] == 'noun':
                for topic in list(set(v['topics'])):
                    topic = '['+ topic +']'
                    headword = v['headword']
                    for sent_en in v['examples']:
                        topic_sent, masked_sent = handle_examples(headword, 
                                                                  sent_en, 
                                                                  topic, 
                                                                  reserve)
                        if topic_sent and masked_sent:
                            f.write('\t'.join([sent_en, topic_sent, 
                                               masked_sent]) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=bool)
    args = parser.parse_args()
    
    brt_groups_results(args.r)
    

if __name__ == '__main__':
    main()