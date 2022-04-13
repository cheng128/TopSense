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
import streamlit as st
from transformers import pipeline
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util

m = nn.Softmax(dim=0)


# set up the environment variables
st.title("TopSense Prototype")
choice = st.radio("model trained epochs", ['hybrid: highest_10_simple_True_5epochs',
                                           'hybrid: highest_10_simple_True_10epochs',
                                           'hybrid: highest_20_simple_True_5epochs',
                                           'hybrid: highest_20_simple_True_10epochs',
                                           'concat: highest_20_simple_False_10epochs',
                                           'concat: highest_10_simple_False_10epochs'])

model_name = choice.split(':')[-1].replace(' ', '')
directory = choice.split(':')[0]


def is_reserve(model_name):
    return 'True' in model_name


RESERVE = is_reserve(model_name)
# =============================================


# load auxiliary data
@st.cache(allow_output_mutation=True)
def load_cambridge():
    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())

    with open('../data/def2data.pickle', 'rb') as f:
        def2data = pickle.load(f)        

    return word2defs, def2data


@st.cache(allow_output_mutation=True)
def load_map():
    with open('../data/orig_new.json') as f:
        data = json.loads(f.read())
        
    filename = '../data/topic_emb/sentence-t5-xl_topic_embs.pickle'
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
        
    with open('../data/definition_emb.pickle', 'rb') as f:
        def_emb_map = pickle.load(f)
        
    with open('../data/orig_new.json') as f:
        topic_name_map = json.loads(f.read())
    return data, emb_map, filename, def_emb_map, topic_name_map


orig_new_map, emb_map, filename, def_emb_map, topic_name_map = load_map()
word2defs, def2data = load_cambridge()
# =================================================================


# load the model
@st.cache(allow_output_mutation=True)
def load_model(directory='brt', model_name='remap_10epochs'):
    mlm = pipeline('fill-mask',
                  model=f"../model/{directory}/{model_name}",
                  tokenizer="../remap_tokenizer")
    return mlm

mlm = load_model(directory, model_name)
# =================================================================


# load spacy and sentence BERT
@st.cache(allow_output_mutation=True)
def load_spacy_sbert():
    model = SentenceTransformer('sentence-t5-xl')
    spacy_model = spacy.load("en_core_web_sm")
    return model, spacy_model

sbert, spacy_model = load_spacy_sbert()
# =================================================================


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


def preprocess(sentence):
    if RESERVE:
        input_sentence = sentence.replace('[' + targetword + ']', f'{targetword} [MASK]')
    else:
        input_sentence = sentence.replace('[' + targetword + ']', '[MASK]')
    return input_sentence


def collect_token_score(mlm_results):
    token_score = {}
    for idx, r in enumerate(mlm_results):
        if r['token_str'].startswith('['):
            topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
            token_score[topic] = r['score']
            st.markdown(str(idx+1) + '. ' + r['token_str'] + "  " +  str(round(r['score'], 3))[1:])
            st.markdown(f"  -> [{topic_name_map[topic]}]")
    return token_score

def rescal_cos_score(score):
    rescal_score = 1 - np.arccos(min(float(score), 1)) / np.pi
    return rescal_score

def reweight_prob(prob):
    complement_prob = 1 - prob
    weight = torch.sigmoid(torch.tensor(prob))
    reweighted_probs = m(torch.tensor([(prob + weight), complement_prob]))
    return float(reweighted_probs[0])

reweight = 1
topic_only = 0

def show_def(token_score):
    word = lemmatize(targetword)
    guide_def, sense2guideword = add_guideword_to_word_definitions(word)
    def_sent_score = calculate_def_sent_score(guide_def)

    # calculate cosine similarity score between topics and definitions
    # weight_score = {}
    # for sense in guide_def:
    #     confidence, topic_score = calculate_topic_conf_and_score(sense, token_score)
    #     weight_score[sense] = confidence * topic_score + (1-confidence) * def_sent_score[sense]
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

    for sense, score in weight_score.items():
        weight_score[sense] = score / len(token_score)

    result = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    
    st.subheader(f'Possible word sense of "{targetword}"')
    # ans = result[0]
    # st.markdown(ans[0] + '  ' + str(round(ans[1], 3))[1:])
    for idx, ans in enumerate(result):
        guideword = sense2guideword[ans[0]][0]
        sense = sense2guideword[ans[0]][1]
        ch_def = def2data[(word, sense)]['ch_def']
        if guideword:
            st.markdown(str(idx+1) + '. ' + guideword + ' ' + sense + ' ' \
                            + str(round(ans[1], 3))[1:])
            st.markdown('     ' + ch_def)
        else:
            st.markdown(str(idx+1) + '. ' + sense + ' ' + str(round(ans[1], 3))[1:])
            st.markdown('     ' + ch_def) 
    return


def lemmatize(word):
    if len(word.split()) > 1:
        return word
    else:
        try:
            lemma = spacy_model(word)[0].lemma_
            if lemma in word2defs:
                return lemma 
        except:
            lemma = spacy_model(word)[0].text
        return lemma


def add_guideword_to_word_definitions(word):
    definitions = word2defs[word][:]
    guide_def = []
    sense2guideword = {}
    for sense in word2defs[word]:
        data = def2data.get((word, sense), '')
        if data:
            guideword = data['guideword']
            sense_add_guideword = guideword[1:-1] + ' ' + sense
            guide_def.append(sense_add_guideword)
            sense2guideword[sense_add_guideword] = [guideword, sense]
        else:
            guide_def.append(sense)
            sense2guideword[sense] = ['', sense]
    return guide_def, sense2guideword


def calculate_def_sent_score(guide_def):
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


def calculate_topic_conf_and_score(sense, token_score):
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

sentence = st.text_area("Sentence here", help='place target word in the []',
                   value='A man was fishing on the opposite [bank].')
targetword = find_target_word(sentence)
clicked = st.button("Enter")


def main():
    mlm_input_sentence = preprocess(sentence)
    mlm_results = mlm(mlm_input_sentence)
    token_score = collect_token_score(mlm_results)
    show_def(token_score)
    return

        
if clicked:
    main()
