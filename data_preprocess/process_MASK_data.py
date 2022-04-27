""" This program can process brt to cambridge data
into training data.
"""
import json
import spacy
import argparse
from nltk import ngrams
from random import sample
from tqdm import tqdm
from utils import *

nlp = spacy.load("en_core_web_sm")

def write_data(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            f.write('\t'.join(data) + '\n')
            
def brt_groups_results(threshold=0.0, reserve=False, pos_tag='noun'):
    
    read_file_name = f'../data/{threshold}.word_id.topics.examples.json'
    print('read:', read_file_name)
    with open(read_file_name) as f:
        data = json.loads(f.read())
    
    training_filename = f'../data/training_data/{threshold}_{reserve}_{pos_tag}_cambridge.tsv'
    print('save training file:', training_filename)
    
    data_list = []
    
    for k, v in tqdm(data.items()):
        if v['pos'] == pos_tag:
            for topic in list(set(v['topics'])):
                topic = '['+ topic +']'
                headword = v['headword']
                for sent_en in v['examples']:
                    topic_sent, masked_sent = handle_examples(headword, 
                                                              sent_en, 
                                                              topic, 
                                                              reserve,
                                                              pos_tag)
                    if topic_sent and masked_sent:
                        data_list.append([sent_en, topic_sent, masked_sent])
    
    write_data(training_filename, data_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=0.0)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-pos', type=str, default='noun')
    args = parser.parse_args()
    threshold = args.t
    reserve = bool(args.r)
    pos_tag = args.pos
    brt_groups_results(threshold, reserve, pos_tag)
    

if __name__ == '__main__':
    main()