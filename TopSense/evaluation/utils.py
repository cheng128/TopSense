import json
from collections import defaultdict

def parse_argument(args):
    filetype = args.f
    trained_model_name = args.m
    mfs_bool = bool(args.mfs)
    topic_only = bool(args.t)
    reweight = bool(args.w)
    reserve = bool(args.r)
    sentence_only = bool(args.s)
    sbert_name = args.sm
    pos_tag = args.pos
    mark = args.a
    return filetype, trained_model_name, mfs_bool, topic_only, reweight,\
            reserve, sentence_only, sbert_name, pos_tag, mark

def print_info(model, mfs_bool, topic_only, reweight, reserve, 
                sentence_only, sbert_name, pos_tag):
    print('predict topic model:', model)
    print('Is most frequent sense: ', mfs_bool)
    print('Is topic only: ', topic_only)
    print('Is reweight: ', reweight)
    print('Is reserve: ', reserve)
    print('Is sentence_only: ', sentence_only)
    print('Sbert model:', sbert_name)
    print('POS tag:', pos_tag)

def load_ans(filetype):
    filename = f'./TopSense/evaluation/data/{filetype}_sentences_ans.json'
    with open(filename) as f:
        sent2ans = json.loads(f.read())
    return sent2ans

def load_mfs_data(filetype):
    if filetype in ['verb', 'adjective']:
        filename = f'./TopSense/evaluation/data/{filetype}_first_sense.json'
    else:
        filename = './TopSense/evaluation/data/first_sense.json'
    with open(filename) as f:
        mfs_data = json.loads(f.read())
    return mfs_data

def gen_save_filename(sentence_only, mfs_bool, topic_only, reserve, trained_model_name, 
                      filetype, reweight, sbert_name, pos_tag, mark):
    directory = trained_model_name.split('/')[-2]
    model_name = trained_model_name.split('/')[-1]
    
    prefix = f'./TopSense/evaluation/results/{directory}/'
    if sentence_only:
        save_file = f'{prefix}{filetype}_sentence_only_{sbert_name}_{pos_tag}{mark}.tsv'
    elif mfs_bool:
        save_file = f'{prefix}{filetype}_most_frequent_{pos_tag}.tsv'
    elif topic_only:
        save_file = f'{prefix}{model_name}_{filetype}_reweight{reweight}_reserve{reserve}_topicOnly_{sbert_name}_{pos_tag}{mark}.tsv'
    else:
        save_file = f'{prefix}{model_name}_{filetype}_reweight{reweight}_reserve{reserve}_{sbert_name}_{pos_tag}{mark}.tsv'
    print('save file: ', save_file)
    return save_file

def load_data(filetype, mfs_bool):
    
    filename = f'./TopSense/evaluation/data/{filetype}_sentences.json'    
    print('load data: ', filename)
    
    with open(filename) as f:
        evaluation_data = json.loads(f.read())
        
    sent2ans = load_ans(filetype)
    first_sense = load_mfs_data(filetype) if mfs_bool else ''

    return evaluation_data, sent2ans, first_sense 

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