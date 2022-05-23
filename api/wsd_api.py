import json
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import sys
sys.path.append('..')
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

word_sense2chdef_level, orig_new = load_data()

sbert_name = 'sentence-t5-xl'
noun_trained_model_name = '../TopSense/model/hybrid/wiki_reserve_new_20_True_4epochs_1e-05' 
# trained_model_name = '../TopSense/model/hybrid/verb_noun_wiki_all_True_15epochs_1e-05'
verb_trained_model_name = '../TopSense/model/hybrid/bootstrap_verb_0.6_all_True_3epochs_1e-05'
tokenizer_name = '../TopSense/tokenizer_casedFalse'
reserve = True
reweight = True
topic_only = False
sentence_only = False

DATA = Data(sbert_name, '../TopSense/data', 'cpu')
NOUN_DISAMBIGUATOR = Disambiguator(DATA, noun_trained_model_name, tokenizer_name,
                reserve, sentence_only, reweight, topic_only)
VERB_DISAMBIGUATOR = Disambiguator(DATA, verb_trained_model_name, tokenizer_name,
                reserve, sentence_only, reweight, topic_only)


class WSDRequest(BaseModel):
    sentence: dict

def gen_store_data(token_scores, ranked_senses, lemma_word):
    topics_data = []
    sorted_topics = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)

    for topic, score in sorted_topics:
        topics_data.append({'text': topic.capitalize(), 
                            'class': orig_new[topic],
                            'score': str(round(score, 2))[1:]})

    output = []
    for sense, score in ranked_senses:
        sense_data = word_sense2chdef_level[lemma_word][sense]
        en_def = sense_data['en_def']
        ch_def = sense_data['ch_def']
        level = sense_data['level'] 
        guideword = sense_data['guideword']
        score = str(1) if score == 1 else str(round(score, 2))[1:]
        output.append({'en_def': en_def,
                       'ch_def': ch_def,
                       'level': level, 
                       'score': score,
                       'guideword': guideword})
    return topics_data, output

@app.post("/api/wsd")
def wsd_sentence(item: WSDRequest):

    sentence = item.sentence
    tokens = sentence['tokens']
    input_sentence = ' '.join([token['text'] for token in tokens])
    
    target_words = [(idx, token['pos'], token['lemma']) 
                    for idx, token in enumerate(tokens)
                    if token['pos'] in ['NOUN', 'PROPN', 'VERB']]


    for idx, pos, targetword in target_words:
        if pos in ['NOUN', 'PROPN']:
            ranked_senses, masked_sent, token_scores = NOUN_DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                                input_sentence, 
                                                                                                pos,
                                                                                                targetword)
        elif pos in ['VERB']: 
            ranked_senses, masked_sent, token_scores = VERB_DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                                input_sentence, 
                                                                                                pos,
                                                                                                targetword)
        if token_scores and ranked_senses:
            topics_data, senses_data = gen_store_data(token_scores, ranked_senses, targetword)
            data = {'topics': topics_data, 'senses': senses_data}
            tokens[idx]['wsd'] = data 

    return JSONResponse({'sentence': {'id': sentence['id'], 'tokens': tokens}}) 

