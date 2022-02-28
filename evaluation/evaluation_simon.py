import json
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

def load_ans():
    with open('./data/100_sentences_ans.json') as f:
        sent2ans = json.loads(f.read())
    return sent2ans

def load_mfs_data():
    with open('./data/first_sense.json') as f:
        mfs_data = json.loads(f.read())
    return mfs_data

def load_emb():
    filename = '../data/topic_embs.pickle'
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

def load_spacy_sbert():
    model = SentenceTransformer('all-roberta-large-v1')
    spacy_model = spacy.load("en_core_web_sm")
    return model, spacy_model 

def fetch_ans(filetype, sentence, sent2ans):
    if filetype == '100':
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

def rescale_cos_score(score):
    rescale_score = 1 - np.arccos(min(float(score), 1)) / np.pi
    return rescale_score

def process(sent, sbert, spacy_model, targetword, token_score, 
            word2defs, emb_map, def2guideword, reserve, 
            sentence_only, topic_only, reweight):
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
        def_sent_score[sentence_defs[j]]  = rescale_cos_score(cos_scores[0][j].cpu())
    
    # if it is sentence_only
    if sentence_only:
        sorted_score = sorted(def_sent_score.items(), key=lambda x: x[1], reverse=True)
        return sorted_score
    
    # calculate cosine similarity score between topics and definitions
    weight_score = {}
    for sense in guide_def:
        embs = sbert.encode(sense, convert_to_tensor=True)
        max_score = float('-inf')
        max_confidence = float('-inf')
        max_topic = ''
        for topic in token_score:
            if topic in emb_map:
                cosine_scores = util.pytorch_cos_sim(embs, emb_map[topic])
                sorted_score = sorted(cosine_scores[0], reverse=True)
                top3_score = [rescale_cos_score(score) for score in sorted_score[:3]]
                avg_score = sum(top3_score) / len(top3_score)
                if avg_score > max_score:
                    max_score = avg_score
                    max_topic = topic
                    max_confidence = token_score[topic]
                elif avg_score == max_score and token_score[topic] > max_confidence:
                    max_topic = topic
                    max_confidence = token_score[topic]
        
        if reweight:
            confidence =  reweight_prob(token_score[max_topic])
        else:
            confidence =  token_score[max_topic]

        if sentence_only:
            weight_score[sense] = def_sent_score[sense]
        elif topic_only:
            weight_score[sense] = confidence * max_score
        else:
            weight_score[sense] = confidence * max_score + (1-confidence) * def_sent_score[sense]
    
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

def print_info(mfs_bool, topic_only, reweight, reserve, sentence_only):
    print('Is most frequent sense: ', mfs_bool)
    print('Is topic only: ', topic_only)
    print('Is sentence_only: ', sentence_only)
    print('Is reweight: ', reweight)
    print('Is reserve: ', reserve)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-m', type=str, default='brt/cambridge_False_10epochs')
    # reserve target word
    parser.add_argument('-r', type=int, default=0)
    # sentence only, default False
    parser.add_argument('-s', type=int, default=0) 
    # most frequent sense
    parser.add_argument('-mfs', type=int, default=0)
    # topic_only
    parser.add_argument('-t', type=int, default=0)
    # reweight
    parser.add_argument('-w', type=int, default=0)
    
    
    args = parser.parse_args()
    filetype = args.f
    model = args.m
    mfs_bool = bool(args.mfs)
    topic_only = bool(args.t)
    reweight = bool(args.w)
    reserve = bool(args.r)
    sentence_only = bool(args.s)
    
    print_info(mfs_bool, topic_only, reweight, reserve, sentence_only)
    
    data, word2defs, def2guideword = load_data(filetype)
    sbert, spacy_model = load_spacy_sbert()
    
    if filetype == '100':
        sent2ans = load_ans()
        
    if mfs_bool:
        first_sense = load_mfs_data()
    
    emb_map = load_emb()
    directory = model.split('/')[0]
    model_name = model.split('/')[-1]

    if sentence_only:
        save_file = f'./results/{directory}/sentence_only.tsv'
    elif mfs_bool:
        save_file = f'./results/{directory}/most_frequent.tsv'
    elif topic_only:
        save_file = f'./results/{directory}/{model_name}_{filetype}_reweight{reweight}_topicOnly_simon_roberta.tsv'
    else:
        save_file = f'./results/{directory}/{model_name}_{filetype}_reweight{reweight}_simon_roberta.tsv'
        
    print('save file: ', save_file)
    if not mfs_bool:
        MLM = load_model(model)
    total_count = 0
    rank_score = 0
    top_one = 0
    
    for targetword, sentences in tqdm(data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype == '100' and not ans:
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
                                  reserve, sentence_only, topic_only, reweight)
                senses = [line[0].replace('\n', '').strip() 
                         for line in results]
                if senses[0] == ans:
                    top_one += 1
                rank_score += 1 / (senses.index(ans) + 1)
                write_data(mfs_bool, sentence_only, input_sent, targetword,
                       ans, senses, topics, save_file)
            total_count += 1
            # write data into file
            
            
                
    with open(save_file, 'a') as f:
        if rank_score:
            f.write('MRR: ' + str(round(rank_score / total_count, 3)) + '\n')
        f.write('Top 1 accuracy: ' + str(round(top_one / total_count, 3)))

if __name__ == '__main__':
    main()