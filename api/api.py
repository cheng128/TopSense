import re
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sys
sys.path.append('..')
from TopSense.disambiguator_class import Disambiguator 
from TopSense.data_class import Data
from TopSense.util import tokenize_processor

app = FastAPI()

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

def load_data():
    # show guideword, level, chinese definitions, etc.
    with open('../TopSense/data/word_sense2chdef_level.json') as f:
        word_sense2chdef_level = json.load(f)

    return word_sense2chdef_level

word_sense2chdef_level = load_data()

def find_target_word(sentence):
    pattern = r'\[[A-Za-z]*\]'
    result = re.search(pattern, sentence)
    if result:
        return result.group()[1:-1]
    else:
        return ''

def find_lemma_targetword(tokens, targetword):
    lemma_word = ''
    for i in tokens:
        if i['text'] == targetword:
            lemma_word = i['lemma']
    return lemma_word

def gen_store_data(ranked_senses, lemma_targetword):
    store_data = []
    for sense, score in ranked_senses:
        data = word_sense2chdef_level[lemma_targetword][sense]
        word_id = data['id']
        store_data.append({'en_def': sense, 'id': word_id, 'score': score})

    return store_data

@app.get("/api/ts")
def return_sense(sentence: str, response_class: JSONResponse):
    targetword = find_target_word(sentence)
    if not targetword:
        return {'message': 'targetword not found.'}
    input_sentence = sentence.replace('[' + targetword + ']', targetword)
    tokens = tokenize_processor(input_sentence)
    targetword = find_lemma_targetword(tokens, targetword)
    ranked_senses, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                      input_sentence, 
                                                                                      'noun',
                                                                                      targetword)
    message = ''
    if ranked_senses and token_scores:
        topics = list(token_scores.keys())
        senses = gen_store_data(ranked_senses, targetword)
        message = {'topics': topics,
                'senses': senses}
    return {'message': message}