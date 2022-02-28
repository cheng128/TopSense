""" This program can process brt to cambridge data
into training data.
"""
import json
import argparse
from random import sample
from tqdm import tqdm
from utils import handle_examples


def write_data(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            f.write('\t'.join(data) + '\n')
    
def brt_groups_results(reserve=False):
    
    with open('../data/remap.word_id.topics.examples.json') as f:
        data = json.loads(f.read())
    
    training_filename = f'../data/training_data/{reserve}_cambridge_train_noun.tsv'
    print('save training file:', training_filename)

    validation_filename = f'../data/validation_data/{reserve}_cambridge_validate_noun.tsv'
    print('save validate file:', validation_filename)
    
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
                        
    data_list = sample(data_list, len(data_list))
    edge = int(len(data_list) * 0.8)
    training_data = data_list[:edge]
    validation_data = data_list[edge:]
    
    write_data(training_filename, training_data)
    write_data(validation_filename, validation_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int)
    args = parser.parse_args()
    reserve = bool(args.r)
    brt_groups_results(reserve)
    

if __name__ == '__main__':
    main()