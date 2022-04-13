# monosemous word -> when handle examples, need to consider POS tag
# not in cam -> len(word.split()) > 1 -> replace directly, else consider POS tag

import json
import spacy
import argparse
from tqdm import tqdm
from process_MASK_data import write_data

nlp = spacy.load("en_core_web_sm")

def load_data():

#     with open('../data/monosemous_sents.json') as f:
#         monosemous_data = json.load(f)

#     with open('../data/not_in_cam_sents.json') as f:
#         not_in_cam = json.load(f)
        
    with open('../data/cam_monosemous_sents.json') as f:
        cam_mono = json.load(f)

    with open('../data/category_num.json') as f:
        category_num = json.load(f)

#     return monosemous_data, not_in_cam, category_num
    return cam_mono, category_num

def handle_examples(headword, sent_en, topic, reserve=False):
    reconstruct = []
    topic_construct = []
    doc = nlp(sent_en)
    
    word = ''
    count = 0
    for token in doc:
        find = False
        if token.text == headword and token.pos_ == 'NOUN':
            word = token.text
            find = True
        elif token.lemma_ == headword and token.pos_ == 'NOUN':
            word = token.lemma_
            find = True

        if find and count == 0:
            if reserve:
                reconstruct.append(word + ' [MASK]')
                topic_construct.append(word + f' {topic}')
            else:
                reconstruct.append('[MASK]')
                topic_construct.append(f'{topic}')
            count += 1
            continue
        
        reconstruct.append(token.text)
        topic_construct.append(token.text)
            
    masked_sent = ' '.join(reconstruct)
    topic_sent = ' '.join(topic_construct)
    
    if '[MASK]' not in masked_sent:
        topic_sent, masked_sent = '', ''
        
    return topic_sent, masked_sent 

def handle_multi_words_example(target_word, sentence, topic, reserve=False):
    target_word = ' ' + target_word + ' '
    masked_sent = ''
    topic_sent = ''
    if target_word in sentence:
        if reserve:
            masked_sent = sentence.replace(target_word, f' {target_word} [MASK] ')
            topic_sent = sentence.replace(target_word, f' {target_word} {topic} ')
        else:
            masked_sent = sentence.replace(target_word, ' [MASK] ')
            topic_sent = sentence.replace(target_word, f' {topic} ')

    return topic_sent, masked_sent  

def process(data_name, category_num, reserve):
    data_list = []

    for word, value in tqdm(data_name.items()):
        word_length = len(word.split())
        if word_length == 1:
            for sent in value['sentences']:
                if 'noun' in value: 
                    for topic in value['noun']:
                        topic_name = '[' + category_num[topic] + ' ' + topic + ']'
                        topic_sent, masked_sent = handle_examples(word, sent, topic_name, reserve)
                        if masked_sent and topic_sent:
                            data_list.append([sent.strip(), topic_sent.strip(), masked_sent.strip()])
        else:
            for sent in value['sentences']: 
                if 'noun' in value: 
                    for topic in value['noun']:
                        topic_name = '[' + category_num[topic] + ' ' + topic + ']'
                        topic_sent, masked_sent = handle_multi_words_example(word, sent, topic_name, reserve)
                        if masked_sent and topic_sent:
                            data_list.append([sent.strip(), topic_sent.strip(), masked_sent.strip()])
    return data_list 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=int, default=0)
    args = parser.parse_args()
    reserve = bool(args.r)
    print('reserve:', reserve)

#     monosemous_data, not_in_cam, category_num = load_data()
    cam_mono, category_num = load_data()
    mono_data_list = process(cam_mono, category_num, reserve)
    write_data(f'../data/training_data/{reserve}_cam_mono_masked_noun.tsv', mono_data_list)

#     monosemous_data_list = process(monosemous_data, category_num, reserve)
#     write_data(f'../data/training_data/{reserve}_monosemous_masked_noun.tsv', monosemous_data_list)
    
#     not_in_cam_data_list = process(not_in_cam, category_num, reserve)    
#     write_data(f'../data/training_data/{reserve}_not_in_cam_masked_noun.tsv', not_in_cam_data_list)

if __name__ == '__main__':
    main()