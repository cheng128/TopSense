import json
import spacy
import pickle
import numpy as np
import streamlit as st
from math import log
from transformers import pipeline
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, util

@st.cache(allow_output_mutation=True)
def load_model(data='brt', epochs='10epochs'):
    if data == 'brt':
        if 'remap' in epochs:
            nlp = pipeline('fill-mask',
                   model=f"../model/brt/{epochs}",
                   tokenizer="../remap_tokenizer")
        else:
            nlp = pipeline('fill-mask',
                           model=f"../model/brt/{epochs}",
                           tokenizer="../topic_tokenizer")
    else:
        nlp = pipeline('fill-mask',
                       model=f"../model/{data}/{epochs}",
                       tokenizer="../topic_tokenizer")
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
def load_map(epochs):
    with open('../data/orig_new.json') as f:
        data = json.loads(f.read())
    if 'remap' in epochs:
        filename = '../data/topic_embs.pickle'
    else:
        filename = '../data/kevin_topic_embs.pickle'
    with open(filename, 'rb') as f:
        emb_map = pickle.load(f)
    return data, emb_map, filename

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
    
    guide_def = [def2guideword.get(sense, '') + ' ' + sense 
                       for sense in word2defs[word]]
    # sentence and definitions
    sentence_defs = [sent.replace(f'[{targetword}]', targetword)]
    sentence_defs.extend(guide_def)
    embs = sbert.encode(sentence_defs, convert_to_tensor=True)
    
    # calculate cosine similarity score between sentence and definitions
    cos_scores = util.pytorch_cos_sim(embs, embs)
    defs_score = {}

    for j in range(1, len(cos_scores)):
        defs_score[sentence_defs[j]]  = 1 - np.arccos(min(float(cos_scores[0][j].cpu()), 1)) / np.pi
    
    # calculate cosine similarity score between topics and definitions
    topic_defs = {}
    definitions = word2defs[word][:] 
    embs = sbert.encode(definitions, convert_to_tensor=True)
    for topic in token_score:     
        if topic in emb_map:
            cosine_scores = util.pytorch_cos_sim(embs, emb_map[topic])
            pairs = []
            edge = len(definitions)
            for i in range(edge):
                for j in range(len(emb_map[topic])):
                    pairs.append({'index': [i, j], 'score': 1 - np.arccos(min(float(cosine_scores[i][j].cpu()), 1)) / np.pi})
            
            pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
            highest_idx = pairs[0]['index'][0]
            topic_defs[topic] = [definitions[highest_idx], pairs[0]['score']]
    
    result = defaultdict(list)
    for topic in token_score:
        if topic in topic_defs:
            sense = topic_defs[topic][0]
            sense = def2guideword.get(sense, '') + ' ' + sense
            result[sense].append(token_score[topic] * topic_defs[topic][1] + (1-token_score[topic]) * defs_score[sense])

    
    for k, v in result.items():
        result[k] = sum(v) / len(v)
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)

    st.subheader(f'Possible word sense of "{targetword}"')
    
    for idx, ans in enumerate(result):
        st.markdown(str(idx+1) + '. ' + ans[0] + ' ' + str(round(ans[1], 3)))

#     topics_defs = list(token_score.keys())
#     edge = len(topics_defs)
#     # topics and definitions
#     topics_defs.extend(definitions)
#     embeddings = sbert.encode(topics_defs, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)  

#     # Find the pairs with the highest cosine similarity scores
#     pairs = []
#     for i in range(len(cosine_scores)-1):
#         for j in range(i+1, len(cosine_scores)):
#             if i < edge and j > edge - 1:
# #                 st.markdown(topics_defs[i] + ' ' + topics_defs[j] + str(cosine_scores[i][j]))
#                 predict_score = token_score[topics_defs[i]]
#                 pairs.append({'index': [i, j], 'score': predict_score * cosine_scores[i][j]})
# #                 pairs.append({'index': [i, j], 'score': ((1/(i+1)) * cosine_scores[i][j].cpu().numpy() +\
# #                                                          defs_score[topics_defs[j]]) / 2})
    
#     #Sort scores in decreasing order
#     pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

#     senses = {}
#     for pair in pairs:
#         i, j = pair['index']
#         if i < edge and topics_defs[i] not in senses and j > edge - 1:
#             senses[topics_defs[i]] = topics_defs[j]
    
#     st.markdown(senses)
#     map2topic = defaultdict(list)
#     for k, v in senses.items():
#         map2topic[v].append(k)

#     vote = Counter(senses.values())
#     vote = sorted(vote.items(), key=lambda x: x[1], reverse=True)

#     st.subheader(f'Possible word sense of "{targetword}"')
#     ans_sense = vote[0][0]
#     category_vote = Counter(['[' + cat_map[i].upper() + ']' for i in map2topic[ans_sense]])
#     category_vote = sorted(category_vote.items(), key=lambda x: x[1], reverse=True)
    
#     if def2guideword[ans_sense]:
#         st.markdown('[' + def2guideword[ans_sense] + '] ' + ans_sense)
#     else:
#         st.markdown(ans_sense)
        
#     st.subheader(f'BRT category:')
#     idx = 1
#     total = sum([i[1] for i in category_vote])
#     for i in category_vote:                
#         st.markdown(str(idx) + '. ' + i[0] + ' ' + str(round(i[1]/total, 3))[1:])
#         idx += 1


st.title("MLM4Topic Prototype")
choice = st.radio("model trained epochs", ['brt: 10epochs', 'brt: remap_5epochs', 'brt: remap_10epochs'])
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
    input_sent = sent.replace('[' + targetword + ']', '[MASK]')
    results = nlp(input_sent)
    orig_new_map, emb_map, filename = load_map(epochs)
    st.markdown(filename)
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
    
                  
                  