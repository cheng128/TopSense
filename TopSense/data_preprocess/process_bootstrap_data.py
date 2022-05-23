import json
import argparse
from tqdm import tqdm
import sys
sys.path.append("..")
from util import gen_masked_sent, tokenize_processor

def load_data():
    with open('../data/bootstrap_word_id.topics.examples.json') as f:
        data = json.load(f)
        
    return data

def write_data(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            f.write('\t'.join(data) + '\n')
            
def brt_groups_results(threshold=0.0, reserve=False, pos_tag='noun'):
    data = load_data()
    training_filename = f'../data/training_data/{threshold}_{reserve}_{pos_tag}_bootstrap_top3.tsv'
    print('save training file:', training_filename)
    
    data_list = []
    
    for k, v in tqdm(data.items()):
        if v['pos'] == pos_tag:
            for topic in list(set(v['topics'])):
                topic = '['+ topic +']'
                headword = v['headword']
                for sent_en, value in v['examples'].items():
                    sent_tokens = tokenize_processor(sent_en)
                    masked_sent, topic_sent = gen_masked_sent(sent_tokens,
                                                              pos_tag,
                                                              headword,  
                                                              reserve,
                                                              topic)
                    if topic_sent and masked_sent:
                        data_list.append([sent_en, topic_sent, masked_sent])
                    # process top3 ngram sentences
                    for ngram, ngram_example in list(value['examples'].items())[:3]:
                        for data in ngram_example:
                            if data['score'] > threshold:
                                sent_tokens = tokenize_processor(data['text'])
                                masked_sent, topic_sent = gen_masked_sent(sent_tokens,
                                                                  pos_tag,
                                                                  headword,  
                                                                  reserve,
                                                                  topic)
                                if topic_sent and masked_sent:
                                    data_list.append([sent_en, topic_sent, masked_sent])

    write_data(training_filename, data_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=0.0)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-pos', type=str, default='noun')
    args = parser.parse_args()
    threshold = float(args.t)
    reserve = bool(args.r)
    pos_tag = args.pos
    brt_groups_results(threshold, reserve, pos_tag)
    

if __name__ == '__main__':
    main()