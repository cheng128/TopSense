# best version so far: v4_2ep
# modified: add filetype (100, 200, 300), topic only, etc.
# use new formula 

import sys
import pdb
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

sys.path.append('../')
from CLR4Topic.CLR4Topic.predictors.topic_predictor import TopicPredictor
from CLR4Topic.CLR4Topic.models.topic_classifier import TopicClassifier
from CLR4Topic.CLR4Topic.dataset_readers.example_reader import ExampleReader

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

def load_topic_emb(sbert_model):
    filename = f'../data/topic_emb/{sbert_model}_topic_embs.pickle'
    print('load emb file: ', filename)
        
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
    return emb_map
            
def load_model(model_name):
    predictor = TopicPredictor.from_path(f'../CLR4Topic/trained_models/{model_name}/model.tar.gz',
                                         'topic_predictor')
    
    return predictor

def load_spacy_sbert(sbert_model):
    model = SentenceTransformer(sbert_model)
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

def process(sent, sbert, spacy_model, targetword, token_score, 
            word2defs, emb_map, def2guideword, reweight):
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
        def_sent_score[sentence_defs[j]]  = 1 - np.arccos(min(float(cos_scores[0][j].cpu()), 1)) / np.pi
    
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
                for j in range(len(emb_map[topic])):
                    score = 1 - np.arccos(min(float(cosine_scores[0][j].cpu()), 1)) / np.pi
                    if score > max_score:
                        max_score = score
                        max_topic = topic
                        max_confidence = token_score[topic]
                    elif score == max_score and token_score[topic] > max_confidence:
                        max_topic = topic
                        max_confidence = token_score[topic]

        if reweight:
            confidence =  reweight_prob(token_score[max_topic])
        else:
            confidence =  token_score[max_topic]

        weight_score[sense] = confidence * max_score #+ (1-confidence) * def_sent_score[sense]
    
    result = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    return result

def rescale_cos_score(score):
    rescale_score = 1 - np.arccos(min(float(score), 1)) / np.pi
    return rescale_score

def formula(sent, sbert, spacy_model, targetword, token_score, 
            word2defs, emb_map, def2guideword, topic_only, reweight):
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
    
    # calculate cosine similarity score between topics and definitions
    weight_score = defaultdict(lambda: 0)
    for topic, confidence in token_score.items():
        if topic in emb_map:
            topic_emb = emb_map[topic]
            for sense in guide_def:
                sense_emb = sbert.encode(sense, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(sense_emb, topic_emb)
                sorted_scores = sorted(cosine_scores[0].cpu(), reverse=True)
                top3_scores = [rescale_cos_score(score) for score in sorted_scores[:3]]
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

def handle_examples(nlp, headword, sent_en):
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
            reconstruct.append(f'[{token.text}]')
            count += 1
            continue
        
        reconstruct.append(token.text)
            
    masked_sent = ' '.join(reconstruct)
    return masked_sent 

def write_data(input_sent, targetword,
               ans, senses, topics, save_file):

    if len(topics) != 5:
        topics += [''] * (5 - len(topics))

    write_data = input_sent + '\t' + targetword + '\t' +\
    '\t'.join(topics) + '\t' + ans + '\t' + '\t'.join(senses) + '\n'

    with open(save_file, 'a') as f:
        f.write(write_data)

def print_info(reweight):
    print('Is reweight: ', reweight)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    # topic_only
    parser.add_argument('-t', type=str)
    # reweight
    parser.add_argument('-w', type=int, default=0)
    # sbert model
    parser.add_argument('-sm', type=str, default='all-roberta-large-v1')
    # model version
    parser.add_argument('-v', type=str, default='v4_2ep')
    
    args = parser.parse_args()
    filetype = args.f
    topic_only = bool(args.t)
    reweight = bool(args.w)
    sbert_model = args.sm
    topic_model_name = args.v
    
    print_info(reweight)
    
    data, word2defs, def2guideword = load_data(filetype)
    sbert, spacy_model = load_spacy_sbert(sbert_model)
    
    if filetype in ['100', '200', '300']:
        sent2ans = load_ans(filetype)
    
    emb_map = load_topic_emb(sbert_model)

    save_file = f'./results/clr/{topic_model_name}_{filetype}_{reweight}_topic{topic_only}_formula_roberta.tsv'
        
    print('save file: ', save_file)
    predictor = load_model(topic_model_name)
    total_count = 0
    rank_score = 0
    top_one = 0
    
    for targetword, sentences in tqdm(data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype in ['100', '200', '300'] and not ans:
                continue

            # I need to go to the [bank] .
            input_sent = handle_examples(spacy_model, targetword, sent)
            results = predictor(input_sent)
            token_score = {}
            for idx, r in enumerate(results):
                if r['token_str'].startswith('['):
                    topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
                    token_score[topic.lower()] = r['score']
                continue
            topics = list(token_score.keys())

            results = formula(sent, sbert, spacy_model, targetword, token_score, 
                        word2defs, emb_map, def2guideword, topic_only, reweight)
            senses = [line[0].replace('\n', '').strip() 
                        for line in results]
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