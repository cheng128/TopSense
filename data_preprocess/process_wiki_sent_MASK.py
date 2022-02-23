import json
import random
import argparse
from tqdm import tqdm
from utils import handle_examples

def load_data(num):
    filename = f'../data/wiki/{num}-t5-xl_wiki_word_id2sents.json'
#     filename = f'../data/{num}-sense_to_gbook_examples.json'
    print('read:', filename)
    with open(filename) as f:
        word_id2sents = json.loads(f.read())        
    
    with open('../data/remap.word_id.topics.examples.json') as f:
        word_id2topics = json.loads(f.read())
    
    return word_id2sents, word_id2topics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int)
    parser.add_argument('-n', type=str)
    
    args = parser.parse_args()
    reserve = bool(args.r)
    directory = 'reserve' if reserve else 'no_reserve'
    num = args.n
    word_id2sents, word_id2topics = load_data(num)
    
    filename = f'../data/training_data/{directory}/{num}-{reserve}-t5-xl_wiki_masked.tsv'
#     filename = f'../data/training_data/{num}-gbooks.tsv'
    print('save: ', filename)
    with open(filename, 'a') as f:
        for key, value in tqdm(word_id2sents.items()):
            if key in word_id2topics:
                word_data = word_id2topics[key]
                if word_data['pos'] == 'noun':
                    topics = word_data['topics']
                    headword = value[0]
                    sentences = value[-1]
                    for topic in list(set(topics)):
                        topic = '['+ topic +']'
                        for sent_en in sentences:
                            sent_en = sent_en.replace('\n', '')
                            topic_sent, masked_sent = handle_examples(headword, 
                                                                      sent_en, 
                                                                      topic,
                                                                      reserve)
                            if topic_sent and masked_sent:
                                f.write('\t'.join([sent_en, topic_sent, 
                                                   masked_sent]) + '\n')
                else:
                    print(key)
if __name__ == '__main__':
    main()