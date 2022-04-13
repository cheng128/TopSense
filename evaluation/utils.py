import os
import pdb
import json
import pickle
import spacy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

m = nn.Softmax(dim=0)

# ============== parse argument ===============
def parse_argument(args):
    filetype = args.f
    model = args.m
    mfs_bool = bool(args.mfs)
    topic_only = bool(args.t)
    reweight = bool(args.w)
    reserve = bool(args.r)
    sentence_only = bool(args.s)
    sbert_model = args.sm
    return filetype, model, mfs_bool, topic_only, \
            reweight, reserve, sentence_only, sbert_model


# ============== decide save file name ==================
def gen_save_filename(sentence_only, mfs_bool, topic_only, reserve,
                       topic_model_name, filetype, reweight,
                        sbert_model, postfix):
    directory = '_'.join(topic_model_name.split('/')[:-1])
    model_name = topic_model_name.split('/')[-1]
    
    prefix = f'./results/{directory}/'
    if sentence_only:
        save_file = f'{prefix}{filetype}_sentence_only_{sbert_model}.tsv'
    elif mfs_bool:
        save_file = f'{prefix}{filetype}_most_frequent.tsv'
    elif topic_only:
        save_file = f'{prefix}{model_name}_{filetype}_reweight{reweight}_reserve{reserve}_topicOnly_{sbert_model}_{postfix}.tsv'
    else:
        save_file = f'{prefix}{model_name}_{filetype}_reweight{reweight}_reserve{reserve}_{sbert_model}_{postfix}.tsv'
    print('save file: ', save_file)
    return save_file

# ============ subfunctions in process function ================
def gen_token_scores(mlm_results):
    token_score = {}
    for idx, r in enumerate(mlm_results):
        if r['token_str'].startswith('['):
            topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
            token_score[topic.lower()] = r['score']
        continue
    return token_score

def gen_guide_def(spacy_model, targetword, word2defs, def2guideword):
    if len(targetword.split()) > 1 :
        word = targetword
    else:
        try:
            word = spacy_model(targetword)[0].lemma_
        except:
            word = spacy_model(targetword)[0].text
        
    definitions = word2defs[word][:]
    # guide_def = []
    # for sense in definitions:
    #     data = def2guideword.get((word, sense), '')
    #     if data:
    #         guideword = data['guideword']
    #         sense_add_guideword = guideword[1:-1] + ' ' + sense
    #         guide_def.append(sense_add_guideword)
    #     else:
    #         guide_def.append(sense)
    # return guide_def
    return definitions

def calculate_def_sent_score(sent, guide_def, SBERT):
    sentence_defs = [sent]
    sentence_defs.extend(guide_def)
    embs = SBERT.encode(sentence_defs, convert_to_tensor=True)

    cos_scores = util.pytorch_cos_sim(embs, embs)

    def_sent_score = {}
    for j in range(1, len(cos_scores)):
        def_sent_score[sentence_defs[j]]  = rescale_cos_score(cos_scores[0][j].cpu())
    return def_sent_score

# ============ MAIN FUNCTION ================

def process_evaluation_data(evaluation_data, filetype, sent2ans, first_sense,
                            mfs_bool, topic_only, reweight, reserve, sentence_only, 
                            save_file, 
                            MLM, SBERT, spacy_model, 
                            word2defs, emb_map, def2guideword,
                            process):
    total_count, rank_score, top_one = 0, 0, 0
    for targetword, sentences in tqdm(evaluation_data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype in ['100', '200', '300'] and not ans:
                continue
                
            # if it is most frequent sense
            if mfs_bool:
                mfs_sense = first_sense[targetword]
                if mfs_sense == ans:
                    top_one += 1
                senses = [mfs_sense]
                write_data(mfs_bool, sentence_only, sent, targetword, 
                           ans, senses, [], save_file)
            else:
                input_sent = handle_examples(spacy_model, targetword, sent, reserve)
                mlm_results = MLM(input_sent)
                token_score = gen_token_scores(mlm_results)
                topics = list(token_score.keys())

                results = process(sent, SBERT, spacy_model, targetword, 
                                  token_score, word2defs, emb_map, def2guideword, 
                                  reserve, sentence_only, topic_only, reweight)
                senses = [line[0].replace('\n', '').strip() 
                         for line in results]
                
                # if not senses:
                #     write_data(mfs_bool, sentence_only, input_sent, targetword,
                #                ans, senses, topics, save_file)
                #     total_count += 1
                #     continue
                if ans.endswith(senses[0]):
                    top_one += 1
                for idx, sense in enumerate(senses):
                    if ans.endswith(sense):
                        rank_score += 1 / (idx + 1)
                # if senses[0] == ans:
                #     top_one += 1
                
                # if ans in senses:
                #     rank_score += 1 / (senses.index(ans) + 1)
                write_data(mfs_bool, sentence_only, input_sent, targetword,
                       ans, senses, topics, save_file)
            total_count += 1

    write_score(save_file, rank_score, top_one, total_count)

# ============= load evaluation data, words2defs, def2guideword data ==============
def load_data(filetype):
    if filetype == 'mix':
        filename = './data/mix_sample.json'
    elif filetype == 'voa':
        filename = './data/voa_sample.json'
    else:
        filename = f'./data/{filetype}_sentences.json'
        
    print('load data: ', filename)
    with open(filename) as f:
        data = json.loads(f.read())
    
    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())
        
    with open('../data/def2data.pickle', 'rb') as f:
        def2guideword = pickle.load(f) 

    return data, word2defs, def2guideword

def load_ans(filetype):
    with open(f'./data/{filetype}_sentences_ans.json') as f:
        sent2ans = json.loads(f.read())
    return sent2ans

def load_mfs_data():
    with open('./data/first_sense.json') as f:
        mfs_data = json.loads(f.read())
    return mfs_data

def load_topic_emb(model_name):
    filename = f'../data/topic_emb/{model_name}_topic_embs.pickle'
    print('load emb file: ', filename)
        
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)

    return emb_map
            
def load_model(model):
    model_path = f"../model/{model}"

    tokenizer = "../remap_tokenizer"
    nlp = pipeline('fill-mask',
           model=model_path,
           tokenizer=tokenizer)
    return nlp

def load_spacy_sbert(model_name):
    model = SentenceTransformer(model_name)
    spacy_model = spacy.load("en_core_web_sm")
    return model, spacy_model 

def fetch_ans(filetype, sentence, sent2ans):
    if filetype in ['100', '200', '300']:
        sent = sentence[0]
        ans = sent2ans.get(sent, '')
    else:
        sent, ans = sentence, ''
    return sent, ans

# =============== calculation functions ==============

def reweight_prob(prob):
    complement_prob = 1 - prob
    weight = torch.sigmoid(torch.tensor(prob))
    reweighted_probs = m(torch.tensor([(prob + weight), complement_prob]))
    return float(reweighted_probs[0])

def rescale_cos_score(score):
    rescale_score = 1 - np.arccos(min(float(score), 1)) / np.pi
    return rescale_score

# =============== replace targetword in sentence ================

def handle_examples(nlp, headword, sent_en, reserve=False):
    reconstruct = []
    doc = nlp(sent_en)
    
    word = ''
    count = 0
    for token in doc:
        find = False
        if token.text.lower() == headword:
            word = token.text
            find = True
        elif token.lemma_.lower() == headword:
            word = token.lemma_
            find = True

        if find and count == 0:
            if reserve:
                reconstruct.append(word + ' [MASK]')
            else:
                reconstruct.append('[MASK]')
            count += 1
            continue
        
        reconstruct.append(token.text)
            
    masked_sent = ' '.join(reconstruct)
    return masked_sent 

# ================== write data, print info ================== 
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

def print_info(model, mfs_bool, topic_only, reweight, reserve, 
                sentence_only, sbert_model):
    print('predict topic model:', model)
    print('Is most frequent sense: ', mfs_bool)
    print('Is topic only: ', topic_only)
    print('Is reweight: ', reweight)
    print('Is reserve: ', reserve)
    print('Is sentence_only: ', sentence_only)
    print('Sbert model:', sbert_model)

def write_score(save_file, rank_score, top_one, total_count):
    with open(save_file, 'a') as f:
        if rank_score:
            f.write('MRR: ' + str(rank_score / total_count) + '\n')
        f.write('Top 1 accuracy: ' + str(top_one / total_count))