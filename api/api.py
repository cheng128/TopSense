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
DISAMBIGUATOR = Disambiguator(DATA, trained_model_name, tokenizer_name,
                reserve, sentence_only, reweight, topic_only)

def find_target_word(sentence):
    pattern = r'\[[A-Za-z]*\]'
    result = re.search(pattern, sentence)
    targetword = result.group()[1:-1]
    return targetword

@app.get("/api/ts")
def return_sense(sentence: str, response_class: JSONResponse):
    targetword = find_target_word(sentence)
    input_sentence = sentence.replace('[' + targetword + ']', targetword)
    tokens = tokenize_processor(input_sentence)
    ranked_senses, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                input_sentence, 
                                                                                'noun',
                                                                                targetword)

    topics = list(token_scores.keys())
    message = {'topics': topics,
              'senses': ranked_senses}
    return {'message': message}