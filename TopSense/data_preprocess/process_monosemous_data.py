import json
import argparse
from tqdm import tqdm
import sys
sys.path.append('..')
from util import tokenize_processor, gen_masked_sent

def load_data(threshold, number):
    with open(f'../data/wiki/{number}_wikipedia_monosemous.json') as f:
        monosemous_data = json.load(f)

    filename = f'../data/{threshold}.word_id.topics.examples.json'
    print('load:', filename)
    with open(filename) as f:
        word_id2topics = json.load(f)

    return monosemous_data, word_id2topics

def write_data(filename, data_list):
    with open(filename, 'w') as f:
        for data in data_list:
            f.write('\t'.join(data) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=0.0)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-n', type=str, default='all')
    parser.add_argument('-pos', type=str, default='noun')
    args = parser.parse_args()
    threshold = args.t
    reserve = bool(args.r)
    number = args.n
    pos_tag = args.pos
    
    directory = 'reserve' if reserve else 'no_reserve'

    monosemous_data, word_id2topics = load_data(threshold, number)

    data_list = []

    for headword, value in tqdm(monosemous_data.items()):
        for pos, data in value.items():    
            if pos == pos_tag:
                word_id = data['id']
                examples = data['examples']
                word_id_data = word_id2topics.get(word_id, [])
                if word_id_data:
                    assert(word_id_data['pos'] == pos)
                    for topic in list(set(word_id_data['topics'])):
                        topic = '['+ topic +']'
                        for example in examples:
                            example = example.replace('\n', ' ')
                            tokens = tokenize_processor(example)
                            masked_sent, topic_sent = gen_masked_sent(tokens, pos, headword, reserve, topic)
                            if topic_sent and masked_sent:
                                data_list.append([example, topic_sent, masked_sent])

    filename = f'../data/training_data/{directory}/{threshold}_{number}_wikipedia_monosemous_{pos_tag}_{reserve}.tsv'
    write_data(filename, data_list)

    return

if __name__ == '__main__':
    main()
