import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from utils import *

app = FastAPI()
model_name = '20_False_10epochs'
MLM = load_model('concat', model_name)
RESERVE = is_reserve(model_name)

@app.get("/api/ts")
def return_sense(sentence: str, response_class: JSONResponse):
    targetword = find_target_word(sentence)
    mlm_input_sentence = preprocess(targetword, sentence, RESERVE)
    mlm_results = MLM(mlm_input_sentence)
    token_score, topics = collect_token_score(mlm_results)
    sorted_senses = sort_sense(targetword, token_score, sentence)
    message = {'topics': topics,
              'senses': sorted_senses}
#     message = {'succeed'}
    return {'message': message}