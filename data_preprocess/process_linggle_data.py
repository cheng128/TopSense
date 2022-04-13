import json
import pdb
import argparse
from tqdm import tqdm
from utils import handle_examples

def load_data(num):
    with open(f'../data/gbook/{num}_sense_to_gbook_examples.json') as f:
        sense2examples = json.loads(f.read())

    with open('../data/remap.word_id.topics.examples.json') as f:
        id2topics = json.loads(f.read())
    
    return sense2examples, id2topics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int)
    parser.add_argument('-n', type=str)
    
    args = parser.parse_args()
    reserve = bool(args.r)
    directory = 'reserve' if reserve else 'no_reserve'
    num = args.n
    sense2examples, id2topics = load_data(num)

    filename = f'../data/training_data/{directory}/{num}_gbook_{reserve}_noun.tsv'

    print('reserve: ', reserve) 
    print('filename: ', filename)

    training_data = []
    for word_id, sentences in tqdm(sense2examples.items()):
        if word_id in id2topics:
            topics_data = id2topics[word_id]
            if topics_data['pos'] == 'noun':
                for topic in list(set(topics_data['topics'])):
                    topic = '['+ topic +']'
                    headword = topics_data['headword']
                    for sent in sentences:
                        topic_sent, masked_sent = handle_examples(headword, 
                                                              sent, 
                                                              topic, 
                                                              reserve)
                        if topic_sent and masked_sent:
                            training_data.append([sent, topic_sent,masked_sent])
                            
    with open(filename, 'a') as f:
        for data in training_data:
            f.write('\t'.join(data) + '\n')
            
    return

if __name__ == '__main__':
    main()
    