import json
import pickle
import spacy
import argparse
import numpy as np
from math import log
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

def load_data(filetype):
    if filetype == 'mix':
        filename = './data/mix_sample.json'
    elif filetype == 'voa':
        filename = './data/voa_sample.json'
    elif filetype == '300':
        filename = './data/300_sentences.json'
        
    print('load data: ', filename)
    with open(filename) as f:
        data = json.loads(f.read())
    
    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())
        
    with open('../data/def2guide.json') as f:
        def2guideword = json.loads(f.read())
    
    return data, word2defs, def2guideword

def load_emb(model):
    if 'remap' in model:
        filename = '../data/topic_embs.pickle'
    else:
        filename = '../data/kevin_topic_embs.pickle'
    print('load emb file: ', filename)
        
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
    return emb_map
            
def load_model(model):
    model_path = f"../model/{model}"
    print('model: ', model_path)
    if 'remap' in model:
        tokenizer = "../remap_tokenizer"
    else:
        tokenizer = "../topic_tokenizer"
        
    nlp = pipeline('fill-mask',
           model=model_path,
           tokenizer=tokenizer)

    return nlp

def load_spacy_sbert():
    model = SentenceTransformer('all-roberta-large-v1')
    spacy_model = spacy.load("en_core_web_sm")
    return model, spacy_model 

def process(sent, sbert, spacy_model, targetword, 
            token_score, word2defs, emb_map, def2guideword):
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
        confidence =  token_score[max_topic]
#         weight_score[sense] = confidence * max_score + (1-confidence) * def_sent_score[sense]
        weight_score[sense] = def_sent_score[sense]
    
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-m', type=str)
    parser.add_argument('-r', type=bool)
    
    args = parser.parse_args()
    filetype = args.f
    model = args.m
    reserve = args.r
    
    data, word2defs, def2guideword = load_data(filetype)
    sbert, spacy_model = load_spacy_sbert()
    
    emb_map = load_emb(model)
    model_file_name = model.replace('/', '_')
    save_file = f'./results/{model_file_name}_{filetype}_{reserve}.tsv'
    print('save file: ', save_file)
    
    MLM = load_model(model)
    
    for targetword, sentences in tqdm(data.items()):
        for sent in sentences:
            sent = sent.strip()
            input_sent = handle_examples(spacy_model, targetword, sent, reserve)
            results = MLM(input_sent)
            token_score = {}
            for idx, r in enumerate(results):
                if r['token_str'].startswith('['):
                    topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
                    token_score[topic] = r['score']

            topics = list(token_score.keys())
            confidence = process(sent, sbert, spacy_model, targetword, 
                                 token_score, word2defs, emb_map, def2guideword)
            sense = [line[0].replace('\n', '') for line in confidence]

            with open(save_file, 'a') as f:
                f.write(input_sent + '\t' + targetword + '\t' +\
                        '\t'.join(topics) + '\t' + '\t'.join(sense) + '\n')

if __name__ == '__main__':
    main()