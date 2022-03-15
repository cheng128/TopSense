import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdb
from utils import *

def load_data():
    with open('../data/word_sense2chdef_level.json') as f:
       word_sense2chdef_level = json.load(f)
    return word_sense2chdef_level

app = FastAPI()
model_name = 'hybrid_True_10epochs_base_concat'
MLM = load_model('hybrid', model_name)
RESERVE = is_reserve(model_name)
word_sense2chdef_level = load_data()

class WSDRequest(BaseModel):
    sentence: str = ""
    lemma_word: str = ""


@app.post("/api/ts/")
def return_sense(item: WSDRequest):
    sentence = item.sentence
    lemma_word = item.lemma_word
    targetword = find_target_word(sentence)
    is_noun = check_is_noun(lemma_word)
    if not is_noun:
        raise HTTPException(status_code=400)

    mlm_input_sentence = preprocess(targetword, sentence, RESERVE)
    mlm_results = MLM(mlm_input_sentence)
    token_score, topics = collect_token_score(mlm_results)
    sorted_senses = sort_sense(targetword, token_score, sentence, RESERVE)
    
    output = []
    try:
        for sense, _ in sorted_senses:
            en_def = word_sense2chdef_level[lemma_word][sense]['en_def']
            ch_def = word_sense2chdef_level[lemma_word][sense]['ch_def']
            level = word_sense2chdef_level[lemma_word][sense]['level'] 
            output.append({'en_def': en_def,
                        'ch_def': ch_def,
                        'level': level})
    except:
        raise HTTPException(status_code=400)

    return JSONResponse({'results': output})