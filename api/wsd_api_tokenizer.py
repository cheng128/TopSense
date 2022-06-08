import json
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

import sys
sys.path.append('../')
from TopSense.disambiguator_class import Disambiguator
from TopSense.data_class import Data

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
    # show guideword, level, chinese definitions, etc.
    with open('../TopSense/data/word_sense2chdef_level.json') as f:
        word_sense2chdef_level = json.load(f)

    # show super group 
    with open('../TopSense/data/orig_new.json') as f:
        orig_new = json.load(f)
    return word_sense2chdef_level, orig_new

def load_tokenizer_map():
    with open('../TopSense/data/all_tokenizer_map.json') as f:
        tokenizer_map = json.load(f)
    return tokenizer_map

word_sense2chdef_level, orig_new = load_data()
tokenizer_map = load_tokenizer_map()

sbert_name = 'sentence-t5-xl'
trained_model_name = '../TopSense/model/hybrid/wiki_reserve_new_20_True_4epochs_1e-05' 
tokenizer_name = '../TopSense/tokenizer_casedFalse'
reserve = True
reweight = True
topic_only = False
sentence_only = False

DATA = Data(sbert_name, '../TopSense/data')
word2pos_defs, topic_embs, sense_examples_embs = DATA.load_data()
sbert_model = DATA.load_sbert_model()
DISAMBIGUATOR = Disambiguator(word2pos_defs, topic_embs, sense_examples_embs,
                              sbert_model, trained_model_name, tokenizer_name,
                              reserve, sentence_only, reweight, topic_only)

class WSDRequest(BaseModel):
    sentence: dict

def gen_store_data(token_scores, sorted_senses, lemma_word):
    topics_data = []
    sorted_topics = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

    for topic, score in sorted_topics:
        topics_data.append({'text': topic.capitalize(), 
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
    input_sentence = ' '.join([token['text'] for token in tokens])
    target_words = [(idx, token['lemma']) for idx, token in enumerate(tokens) 
                    if token['pos'] in ['NOUN', 'PROPN']]


    for idx, targetword in target_words:

        ranked_senses, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                input_sentence, 
                                                                                tokens[idx]['pos'],
                                                                                targetword)
        if token_scores and ranked_senses:
            topics_data, senses_data = gen_store_data(token_scores, ranked_senses, targetword)
            data = {'topics': topics_data, 'senses': senses_data}
            tokens[idx]['wsd'] = data 

    return JSONResponse({'sentence': {'id': sentence['id'], 'tokens': tokens}}) 

