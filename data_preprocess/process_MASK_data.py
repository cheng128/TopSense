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
            
def brt_groups_results(reserve=False):
    
    read_file_name = '../data/0.6_remap.word_id.topics.examples.json'
    print('read:', read_file_name)
    with open(read_file_name) as f:
        data = json.loads(f.read())
    
    training_filename = f'../data/training_data/0.6_cross_ref_reserve{reserve}_cambridge.tsv'
    print('save training file:', training_filename)
    
    data_list = []
    
    for k, v in tqdm(data.items()):
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
                        data_list.append([sent_en, topic_sent, masked_sent])
    
    write_data(training_filename, data_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int, default=0)
    args = parser.parse_args()
    reserve = bool(args.r)
    brt_groups_results(reserve)
    

if __name__ == '__main__':
    main()