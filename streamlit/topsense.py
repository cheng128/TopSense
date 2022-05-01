import json
import streamlit as st
import torch.nn as nn

softmax = nn.Softmax(dim=0)

st.title("TopSense Prototype")

choice = st.radio("model trained epochs", ['hybrid: wiki_reserve_new_20_True_4epochs_1e-05',
                                           'hybrid: noun_verb_all_True_5epochs_1e-05'])

model_name = choice.split(':')[-1].replace(' ', '')
directory = choice.split(':')[0]

def is_reserve(model_name):
    return 'True' in model_name

RESERVE = is_reserve(model_name)

@st.cache(allow_output_mutation=True)
def load_cambridge():
    with open('../data/word2pos_defs.json') as f:
        word2defs = json.loads(f.read())

    with open('../data/def2data.pickle', 'rb') as f:
        def2data = pickle.load(f)        

    return word2defs, def2data