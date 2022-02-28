import json
import pdb
import pickle
import spacy
import torch
import argparse
import numpy as np
import torch.nn as nn
from math import log
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

m = nn.Softmax(dim=0)

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
        
    with open('../data/def2guide.json') as f:
        def2guideword = json.loads(f.read())

    return data, word2defs, def2guideword

def load_ans(filetype):
    with open(f'./data/{filetype}_sentences_ans.json') as f:
        sent2ans = json.loads(f.read())
    return sent2ans

def load_topic_emb(model_name):
    filename = f'../data/topic_emb/{model_name}_topic_embs.pickle'
    print('load emb file: ', filename)
        
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
    return emb_map
            
def load_model(model):
    model_path = f"../model/{model}"
    print('model: ', model_path)

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
    
def reweight_prob(prob):
    complement_prob = 1 - prob
    weight = torch.sigmoid(torch.tensor(prob))
    reweighted_probs = m(torch.tensor([(prob + weight), complement_prob]))
    return float(reweighted_probs[0])

def rescal_cos_score(score):
    rescal_score = 1 - np.arccos(min(float(score), 1)) / np.pi
    return rescal_score

def process(sent, sbert, spacy_model, targetword, token_score, 
            word2defs, emb_map, def2guideword, reserve, topic_only, reweight):
    try:
        word = spacy_model(targetword)[0].lemma_
    except:
        word = spacy_model(targetword)[0].text
        
    definitions = word2defs[word][:]
    guide_def = [def2guideword.get(sense, '') + ' ' + sense 
                       for sense in word2defs[word]]
    # sentence and definitions
    
    sentence_defs = [sent]
    sentence_defs.extend(guide_def)
    embs = sbert.encode(sentence_defs, convert_to_tensor=True)
    
    # calculate cosine similarity score between sentence and definitions
    cos_scores = util.pytorch_cos_sim(embs, embs)
    def_sent_score = {}

    for j in range(1, len(cos_scores)):
        def_sent_score[sentence_defs[j]]  = rescal_cos_score(cos_scores[0][j].cpu())

    # calculate cosine similarity score between topics and definitions
    weight_score = defaultdict(lambda: 0)
    for topic, confidence in token_score.items():
        if topic in emb_map:
            topic_emb = emb_map[topic]
            for sense in guide_def:
                sense_emb = sbert.encode(sense, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(sense_emb, topic_emb)
                sorted_scores = sorted(cosine_scores[0].cpu(), reverse=True)
                top3_scores = [rescal_cos_score(score) for score in sorted_scores[:3]]
                if reweight:
                    confidence =  reweight_prob(token_score[topic])
                else:
                    confidence =  token_score[topic]
                if topic_only:
                    sense_score = confidence * sum(top3_scores) / len(top3_scores)
                else:   
                    sense_score = confidence * sum(top3_scores) / len(top3_scores) +\
                                     (1 - confidence) * def_sent_score[sense]
                weight_score[sense] += sense_score 
    
    result = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    return result

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

def write_data(input_sent, targetword, ans, senses, topics, save_file):

    if len(topics) != 5:
        topics += [''] * (5 - len(topics))

    write_data = input_sent + '\t' + targetword + '\t' +\
    '\t'.join(topics) + '\t' + ans + '\t' + '\t'.join(senses) + '\n'

    with open(save_file, 'a') as f:
        f.write(write_data)

def print_info(reweight, reserve, topic_only):
    print('Is reweight:', reweight)
    print('Is reserve:', reserve)
    print('Is topic_only:', topic_only)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-m', type=str, default='brt/cambridge_False_10epochs')
    # reserve target word
    parser.add_argument('-r', type=int, default=0)
    # topic_only
    parser.add_argument('-t', type=int, default=0)
    # reweight
    parser.add_argument('-w', type=int, default=0)
    # sbert model
    parser.add_argument('-sm', type=str, default='all-roberta-large-v1')
    
    
    args = parser.parse_args()
    filetype = args.f
    model = args.m
    reweight = bool(args.w)
    reserve = bool(args.r)
    topic_only = bool(args.t)
    sbert_name = args.s
    
    print_info(reweight, reserve, topic_only)
    
    data, word2defs, def2guideword = load_data(filetype)
    sbert, spacy_model = load_spacy_sbert(sbert_name)
    
    if filetype in ['100', '200', '300']:
        sent2ans = load_ans(filetype)
    
    emb_map = load_topic_emb(sbert_name)
    directory = model.split('/')[0]
    model_name = model.split('/')[-1]

    save_file = f'./results/{directory}/{model_name}_{filetype}_reweight{reweight}_topic{topic_only}_formula_{sbert_name}.tsv'
        
    print('save file: ', save_file)
    MLM = load_model(model)
    total_count = 0
    rank_score = 0
    top_one = 0
    
    for targetword, sentences in tqdm(data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype in ['100', '200', '300'] and not ans:
                continue
                
            # if it is most frequent sense
            input_sent = handle_examples(spacy_model, targetword, sent, reserve)
            results = MLM(input_sent)
            token_score = {}
            for idx, r in enumerate(results):
                if r['token_str'].startswith('['):
                    topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
                    token_score[topic] = r['score']
                continue
            topics = list(token_score.keys())

            results = process(sent, sbert, spacy_model, targetword, 
                                token_score, word2defs, emb_map, def2guideword, 
                                reserve, topic_only, reweight)
            senses = [sense[0].replace('\n', '').strip() 
                        for sense in results]
            if senses[0] == ans:
                top_one += 1 
            rank_score += 1 / (senses.index(ans) + 1)
            write_data(input_sent, targetword, ans, senses, topics, save_file)
            total_count += 1
            # write data into file
                
    with open(save_file, 'a') as f:
        if rank_score:
            f.write('MRR: ' + str(round(rank_score / total_count, 3)) + '\n')
        f.write('Top 1 accuracy: ' + str(round(top_one / total_count, 3)))

if __name__ == '__main__':
    main()