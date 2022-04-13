"""
input:
    1. model name
    2. sentence

1. load data
2. load model
3. load spacy, sentence BERT
4. predict:
    1. preprocess sentence: 需要區分 model (reserve or non-reserve)
    2. show definitions: 需要區分 model (reserve or non-reserve)

remap/topic and reserve 是保存在 model name 裡面
"""


import json
import spacy
import pickle
import torch
import numpy as np
import torch.nn as nn
from math import log
from transformers import pipeline
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util

m = nn.Softmax(dim=0)


def is_reserve(model_name):
    return 'True' in model_name

def load_cambridge():
    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())
        
    with open('../data/def2data.pickle', 'rb') as f:
        def2data = pickle.load(f) 
    return word2defs, def2data

def load_map():
    with open('../data/orig_new.json') as f:
        data = json.loads(f.read())
        
    filename = '../data/topic_emb/sentence-t5-xl_topic_embs.pickle'
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
        
    with open('../data/definition_emb.pickle', 'rb') as f:
        def_emb_map = pickle.load(f)
    return data, emb_map, filename, def_emb_map


orig_new_map, emb_map, filename, def_emb_map = load_map()
word2defs, def2data = load_cambridge()

def check_is_noun(word):
    if word in word2defs:
        return True
    else:
        return False

def load_model(directory='brt', model_name='remap_10epochs'):
    mlm = pipeline('fill-mask',
                  model=f"../model/{directory}/{model_name}",
                  tokenizer="../remap_tokenizer")
    return mlm

def load_spacy_sbert():
    model = SentenceTransformer('sentence-t5-xl')
    spacy_model = spacy.load("en_core_web_sm")
    return model, spacy_model

sbert, spacy_model = load_spacy_sbert()

def rescal_cos_score(score):
    rescal_score = 1 - np.arccos(min(float(score), 1)) / np.pi
    return rescal_score

def reweight_prob(prob):
    complement_prob = 1 - prob
    weight = torch.sigmoid(torch.tensor(prob))
    reweighted_probs = m(torch.tensor([(prob + weight), complement_prob]))
    return float(reweighted_probs[0])

# predict topics and perform the WSD
def find_target_word(sent):
    start = False
    end = False
    result = ''
    for i in sent:
        if i == '[':
            start = True
            continue
        if i == ']':
            end = True
        if start and not end:
            result += i
    return result 


def preprocess(targetword, sentence, RESERVE):
    if RESERVE:
        input_sentence = sentence.replace('[' + targetword + ']', f'{targetword} [MASK]')
    else:
        input_sentence = sentence.replace('[' + targetword + ']', '[MASK]')
    return input_sentence


def collect_token_score(mlm_results):
    token_score = {}
    topics = {}
    for idx, r in enumerate(mlm_results):
        if r['token_str'].startswith('['):
            topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
            token_score[topic] = r['score']
            topics[r['token_str']] = r['score']
    return token_score, topics


def sort_sense(targetword, token_score, sentence, RESERVE):
    word = lemmatize(targetword)
    guide_def = add_guideword_to_word_definitions(word)
    def_sent_score = calculate_def_sent_score(guide_def, sentence, targetword)
    
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
                confidence =  reweight_prob(token_score[topic])
                sense_score = confidence * sum(top3_scores) / len(top3_scores) +\
                                    (1 - confidence) * def_sent_score[sense]
                weight_score[sense] += sense_score  

    for sense, score in weight_score.items():
        weight_score[sense] = score / len(token_score)
    
    sorted_sense = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    return sorted_sense


def lemmatize(word):
    try:
        lemma = spacy_model(word)[0].lemma_
    except:
        lemma = spacy_model(word)[0].text
    return lemma


def add_guideword_to_word_definitions(word):

    definitions = word2defs[word][:]
    guide_def = []
    for sense in word2defs[word]:
        data = def2data.get((word, sense), '')
        if data:
            guideword = data['guideword']
            if guideword:
                sense_add_guideword = guideword[1:-1] + ' ' + sense
            else:
                sense_add_guideword = sense
            guide_def.append(sense_add_guideword)
        else:
            guide_def.append(sense)
    return guide_def


def calculate_def_sent_score(guide_def, sentence, targetword):
    # sentence and definitions
    sentence_defs = [sentence.replace(f'[{targetword}]', targetword)]
    sentence_defs.extend(guide_def)
    embs = sbert.encode(sentence_defs, convert_to_tensor=True)
    
    # calculate cosine similarity score between sentence and definitions
    cos_scores = util.pytorch_cos_sim(embs, embs)
    def_sent_score = {}

    for j in range(1, len(cos_scores)):
        def_sent_score[sentence_defs[j]]  = 1 - np.arccos(min(float(cos_scores[0][j].cpu()), 1)) / np.pi
    return def_sent_score


def calculate_topic_conf_and_score(sense, token_score, RESERVE):
    # TODO: replace sbert.encode with def_emb_map
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

    if RESERVE:
        confidence = reweight_confidence(token_score[max_topic])
    else:
        confidence = token_score[max_topic]
    return (confidence, max_score)


def reweight_confidence(confidence):
    complement_confidence = 1 - confidence
    weight = torch.sigmoid(torch.tensor(confidence))
    reweighted_confidence = m(torch.tensor([(confidence + weight), complement_confidence]))
    return float(reweighted_confidence[0])
