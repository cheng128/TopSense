import json
import pickle
import sys
sys.path.append('../')
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from evaluation.evaluation_formula import cal_weighted_score
from evaluation.utils import load_model, load_topic_emb, calculate_def_sent_score
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

def load_data():
    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())

    with open('../data/def2data.pickle', 'rb') as f:
        def2guideword = pickle.load(f) 

    with open('../data/word_sense2chdef_level.json') as f:
        word_sense2chdef_level = json.load(f)
        
    with open('../data/orig_new.json') as f:
        orig_new = json.load(f)
    
    return word2defs, def2guideword, word_sense2chdef_level, orig_new

def load_tokenizer_map():
    with open('../all_tokenizer_map.json') as f:
        tokenizer_map = json.load(f)
        
    return tokenizer_map

sbert_model = 'sentence-t5-xl'
SBERT = SentenceTransformer(sbert_model)
MLM = load_model('hybrid/wiki_reserve_20_True_4epochs_1e-05')
reweight = True
topic_only = False
RESERVE = True
emb_map = load_topic_emb(sbert_model)
word2defs, def2guideword, word_sense2chdef_level, orig_new = load_data()
tokenizer_map = load_tokenizer_map()

class WSDRequest(BaseModel):
    sentence: dict

def gen_guide_def(word):
    definitions = set(word2defs[word][:])
    guide_def = {}
    for sense in definitions:
        data = def2guideword.get((word, sense), '')
        if data['guideword']:
            guideword = data['guideword']
            sense_add_guideword = guideword[1:-1] + ' ' + sense
            guide_def[sense_add_guideword] = guideword
        else:
            guide_def[sense] = ''
    return guide_def

def gen_mask_sent(sent_list, targetword, reserve=True):

    reconstruct = []
    
    word = ''
    count = 0
    for token in sent_list:
        find = False
        if token['lemma'] == targetword:
            find = True

        if find and count == 0:
            if reserve:
                reconstruct.append(token['text'] + ' [MASK]')
            else:
                reconstruct.append('[MASK]')
            count += 1
            continue
        
        reconstruct.append(token['text'])
            
    masked_sent = ' '.join(reconstruct)
    return masked_sent 

def gen_store_data(token_scores, sorted_senses, lemma_word):
    topics_data = []
    sorted_topics = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

    for topic, score in sorted_topics:
        topics_data.append({'text': topic, 
                            'class': orig_new[topic],
                            'score': str(round(score, 2))[1:]})

    output = []
    for sense, score in sorted_senses:
        sense_data = word_sense2chdef_level[lemma_word][sense]
        en_def = sense_data['en_def']
        ch_def = sense_data['ch_def']
        level = sense_data['level'] 
        guideword = sense_data['guideword']
        if score == 1:
            score = str(1)
        else:
            score = str(round(score, 2))[1:]
        output.append({'en_def': en_def,
                    'ch_def': ch_def,
                    'level': level, 
                    'score': score,
                    'guideword': guideword})
    return topics_data, output

def gen_token_scores(mlm_results):
    token_score = {}
    for idx, r in enumerate(mlm_results):
        if r['token_str'].startswith('['):
            try:
                origin_topic = tokenizer_map[str(r['token'])]
                topic = ' '.join(origin_topic.split(' ')[1:])[:-1].lower()
                token_score[topic] = r['score']
            except:
                print(r['token'])
                continue
    return token_score

@app.post("/api/wsd")
def wsd_sentence(item: WSDRequest):
    
    sentence = item.sentence
    tokens = sentence['tokens']
    sent = [token['text'] for token in tokens]
    target_words = [(idx, token['lemma']) for idx, token in enumerate(tokens) 
                    if token['pos'] in ['NOUN', 'PROPN']]

    for idx, targetword in target_words:
        if targetword in word2defs:
            masked_sent = gen_mask_sent(tokens, targetword, RESERVE)
            mlm_results = MLM(masked_sent)
            token_scores = gen_token_scores(mlm_results)
            # guide_def = gen_guide_def(targetword)
            definitions = list(set(word2defs[targetword][:]))
            if len(definitions) > 1:
                def_sent_score = calculate_def_sent_score(sent, definitions, SBERT)
                sorted_senses = cal_weighted_score(token_scores, emb_map, definitions, SBERT,
                                                    reweight, topic_only, def_sent_score)
            else:
                sorted_senses = [[definitions[0], 1]]

            topics_data, senses_data = gen_store_data(token_scores, sorted_senses, targetword)
            data = {'topics': topics_data, 'senses': senses_data}
            tokens[idx]['wsd'] = data 

    return JSONResponse({'sentence': {'id': sentence['id'], 'tokens': tokens}}) 

