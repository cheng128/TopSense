import json
import spacy
import pickle
import torch
import numpy as np
import torch.nn as nn
import streamlit as st
from math import log
from transformers import pipeline
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util

m = nn.Softmax(dim=0)

def reweight_prob(prob):
    complement_prob = 1 - prob
    weight = torch.sigmoid(torch.tensor(prob))
    reweighted_probs = m(torch.tensor([(prob + weight), complement_prob]))
    return float(reweighted_probs[0])

@st.cache(allow_output_mutation=True)
def load_model(data='brt', epochs='remap_10epochs'):
    if data == 'brt':
        if 'remap' in epochs:
            nlp = pipeline('fill-mask',
                   model=f"../model/brt/{epochs}",
                   tokenizer="../remap_tokenizer")
        else:
            nlp = pipeline('fill-mask',
                           model=f"../model/brt/{epochs}",
                           tokenizer="../fast_tokenizer")
    elif data == 'lr_1e-5':
        nlp = pipeline('fill-mask',
                       model=f"../model/{data}/{epochs}",
                       tokenizer="../remap_tokenizer")
    else:
        nlp = pipeline('fill-mask',
                      model=f"../model/{data}/{epochs}",
                      tokenizer="../remap_tokenizer")
    return nlp

@st.cache(allow_output_mutation=True)
def load_spacy_sbert():
    model = SentenceTransformer('all-roberta-large-v1')
    spacy_model = spacy.load("en_core_web_sm")
    return model, spacy_model

@st.cache(allow_output_mutation=True)
def load_cambridge():
    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())
        
    with open('../data/def2guide.json') as f:
        def2guideword = json.loads(f.read())
    return word2defs, def2guideword

@st.cache(allow_output_mutation=True)
def load_map():
    with open('../data/orig_new.json') as f:
        data = json.loads(f.read())
        
    if 'remap' in epochs:
        filename = '../data/topic_embs.pickle'
    else:
        filename = '../data/kevin_topic_embs.pickle'
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
        
    with open('../data/definition_emb.pickle', 'rb') as f:
        def_emb_map = pickle.load(f)
    return data, emb_map, filename, def_emb_map

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
    
def show_def(targetword, token_score, word2defs, def2guideword, cat_map, emb_map):
    try:
        word = spacy_model(targetword)[0].lemma_
    except:
        word = spacy_model(targetword)[0].text
        
    definitions = word2defs[word][:]
    guide_def = [def2guideword.get(sense, '') + ' ' + sense 
                       for sense in word2defs[word]]
    # sentence and definitions
    sentence_defs = [sent.replace(f'[{targetword}]', targetword)]
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
        confidence = reweight_prob(confidence)
        weight_score[sense] = confidence * max_score + (1-confidence) * def_sent_score[sense]
    
    result = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    
    st.subheader(f'Possible word sense of "{targetword}"')
    
    for idx, ans in enumerate(result):
        st.markdown(str(idx+1) + '. ' + ans[0] + ' ' + str(round(ans[1], 3)))
        

st.title("MLM4Topic Prototype")
choice = st.radio("model trained epochs", ['reserve: reserve_remap_10epochs'])
epochs = choice.split(':')[-1].replace(' ', '')
data = choice.split(':')[0]

# targetword = st.text_input('target word', 'banks')
sent = st.text_area("Sentence here", help='place target word in the []',
                   value='In freshwater ecology, [banks] are of interest as the location of riparian habitats. Riparian zones occur along upland and lowland river and stream beds.')
targetword = find_target_word(sent)

nlp = load_model(data, epochs)
sbert, spacy_model = load_spacy_sbert()
clicked = st.button("Enter")

def main():
#     input_sent = sent.replace('[' + targetword + ']', '[MASK]')
    input_sent = sent.replace('[' + targetword + ']', f'{targetword} [MASK]')
    results = nlp(input_sent)
    orig_new_map, emb_map, filename, def_emb_map = load_map()
    word2defs, def2guideword = load_cambridge()
    token_score = {}
    for idx, r in enumerate(results):
        if r['token_str'].startswith('['):
            topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
            token_score[topic] = r['score']
            st.markdown(str(idx+1) + '. ' + r['token_str'] + "  " + str(round(r['score'], 3))[1:])
            
    show_def(targetword, token_score, word2defs, def2guideword, orig_new_map, emb_map)
        
if clicked:
    main()
    
                  
                  