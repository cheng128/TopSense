import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.append("..")
from disambiguator_class import Disambiguator 
from data_class import Data

def parse_argument(args):
    filetype = args.f
    trained_model_name = args.m
    mfs_bool = bool(args.mfs)
    topic_only = bool(args.t)
    reweight = bool(args.w)
    reserve = bool(args.r)
    sentence_only = bool(args.s)
    sbert_name = args.sm
    return filetype, trained_model_name, mfs_bool, topic_only,\
            reweight, reserve, sentence_only, sbert_name

def print_info(model, mfs_bool, topic_only, reweight, reserve, 
                sentence_only, sbert_name):
    print('predict topic model:', model)
    print('Is most frequent sense: ', mfs_bool)
    print('Is topic only: ', topic_only)
    print('Is reweight: ', reweight)
    print('Is reserve: ', reserve)
    print('Is sentence_only: ', sentence_only)
    print('Sbert model:', sbert_name)

def load_ans(filetype):
    with open(f'./data/{filetype}_sentences_ans.json') as f:
        sent2ans = json.loads(f.read())
    return sent2ans

def load_mfs_data():
    with open('./data/first_sense.json') as f:
        mfs_data = json.loads(f.read())
    return mfs_data

def gen_save_filename(sentence_only, mfs_bool, topic_only, reserve, trained_model_name, 
                      filetype, reweight, sbert_name):
    directory = trained_model_name.split('/')[0]
    model_name = trained_model_name.split('/')[-1]
    
    prefix = f'./results/{directory}/'
    if sentence_only:
        save_file = f'{prefix}{filetype}_sentence_only_{sbert_name}.tsv'
    elif mfs_bool:
        save_file = f'{prefix}{filetype}_most_frequent.tsv'
    elif topic_only:
        save_file = f'{prefix}{model_name}_{filetype}_reweight{reweight}_reserve{reserve}_topicOnly_{sbert_name}.tsv'
    else:
        save_file = f'{prefix}{model_name}_{filetype}_reweight{reweight}_reserve{reserve}_{sbert_name}.tsv'
    print('save file: ', save_file)
    return save_file

def load_data(filetype, mfs_bool):
    if filetype == 'mix':
        filename = './data/mix_sample.json'
    elif filetype == 'voa':
        filename = './data/voa_sample.json'
    else:
        filename = f'./data/{filetype}_sentences.json'
        
    print('load data: ', filename)
    with open(filename) as f:
        evaluation_data = json.loads(f.read())

    if filetype in ['100', '200', '300']:
        sent2ans = load_ans(filetype)
    else:
        sent2ans = defaultdict(lambda:' ')

    first_sense = load_mfs_data() if mfs_bool else ''

    return evaluation_data, sent2ans, first_sense 

def fetch_ans(filetype, sentence, sent2ans):
    if filetype in ['100', '200', '300']:
        sent = sentence[0]
        ans = sent2ans.get(sent, '')
    else:
        sent, ans = sentence, ''
    return sent, ans

def process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, sentence_only, 
                      sent, save_filename):

    mfs_sense = first_sense[targetword]
    if mfs_sense == ans:
        top_one += 1
    senses = [mfs_sense]
    write_data(mfs_bool, sentence_only, sent, targetword, 
                ans, senses, [], save_filename)
    return top_one

def write_data(mfs_bool, sentence_only, input_sent, targetword,
               ans, senses, topics, save_file):

    if sentence_only or mfs_bool:
        write_data = input_sent + '\t' + targetword + '\t' +\
        ans + '\t' + '\t'.join(senses) + '\n'
    else:
        if len(topics) != 5:
            topics += [''] * (5 - len(topics))
    
        write_data = input_sent + '\t' + targetword + '\t' +\
        '\t'.join(topics) + '\t' + ans + '\t' + '\t'.join(senses) + '\n'

    with open(save_file, 'a') as f:
        f.write(write_data)

def write_score(save_file, rank_score, top_one, total_count):
    with open(save_file, 'a') as f:
        if rank_score:
            f.write('MRR: ' + str(rank_score / total_count) + '\n')
        f.write('Top 1 accuracy: ' + str(top_one / total_count))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str) # evaluation data: 100, 200, 300
    parser.add_argument('-m', type=str, default='brt/cambridge_False_10epochs')
    parser.add_argument('-r', type=int, default=0)  # reserve target word
    parser.add_argument('-s', type=int, default=0)  # sentence only
    parser.add_argument('-mfs', type=int, default=0) # most frequent sense
    parser.add_argument('-t', type=int, default=0)  # topic_only
    parser.add_argument('-w', type=int, default=0)  # reweight
    parser.add_argument('-sm', type=str, default='sentence-t5-xl') # sbert_name
    args = parser.parse_args()

    assert((args.mfs + args.s + args.t) <= 1)

    filetype, trained_model_name, mfs_bool, topic_only,\
    reweight, reserve, sentence_only, sbert_name = parse_argument(args)

    DATA = Data(sbert_name, '../data')
    DISAMBIGUATOR = Disambiguator(DATA, trained_model_name,
                                  reserve, sentence_only, reweight, topic_only)
    
    print_info(trained_model_name, mfs_bool, topic_only, reweight, 
                reserve, sentence_only, sbert_name)
    
    evaluation_data, sent2ans, first_sense = load_data(filetype, mfs_bool)
    save_filename = gen_save_filename(sentence_only, mfs_bool, topic_only, reserve,
                                    trained_model_name, filetype, reweight, sbert_name)
                                    
    if mfs_bool:
       process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, 
                         sentence_only, sent, targetword, save_filename) 

    total_count, rank_score, top_one = 0, 0, 0
    for targetword, sentences in tqdm(evaluation_data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype in ['100', '200', '300'] and not ans:
                continue
                
            # if it is most frequent sense
            if mfs_bool:
                top_one = process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, 
                                            sentence_only, sent, targetword, save_filename)
            else:
                results, input_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(sentence[0], 
                                                                                           'noun', targetword)
                if not results:
                    results, input_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(sentence[0],
                                                                                               'propn', targetword)
                senses = [line[0].replace('\n', '').strip() for line in results]
                
                if ans.endswith(senses[0]):
                    top_one += 1
                for idx, sense in enumerate(senses):
                    if ans.endswith(sense):
                        rank_score += 1 / (idx + 1)
                topics = list(token_scores.keys())
                write_data(mfs_bool, sentence_only, input_sent, targetword,
                       ans, senses, topics, save_filename)
            total_count += 1

    write_score(save_filename, rank_score, top_one, total_count)

if __name__ == '__main__':
    main()