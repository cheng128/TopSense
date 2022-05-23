import re
import json
import streamlit as st
import sys
sys.path.append('..')
from TopSense.disambiguator_class import Disambiguator 
from TopSense.data_class import Data
from TopSense.util import tokenize_processor

st.title("TopSense Prototype")

choice = st.radio("model trained epochs", ['hybrid: wiki_reserve_new_20_True_4epochs_1e-05',
                                           'hybrid: noun_verb_all_True_5epochs_1e-05'])

model_name = choice.split(':')[-1].replace(' ', '')
directory = choice.split(':')[0]

trained_model_name = f'../TopSense/model/{directory}/{model_name}'
tokenizer_name = '../TopSense/tokenizer_casedFalse'
sbert_name = 'sentence-t5-xl'
reserve = True
reweight = True
topic_only = False
sentence_only = False

@st.cache(allow_output_mutation=True)
def initial_class():
    DATA = Data(sbert_name, '../TopSense/data')
    DISAMBIGUATOR = Disambiguator(DATA, trained_model_name, tokenizer_name,
                    reserve, sentence_only, reweight, topic_only)
    return DATA, DISAMBIGUATOR

DATA, DISAMBIGUATOR = initial_class()

@st.cache(allow_output_mutation=True)
def load_data():
    # show guideword, level, chinese definitions, etc.
    with open('../TopSense/data/word_sense2chdef_level.json') as f:
        word_sense2chdef_level = json.load(f)

    # show super group 
    print('load orig')
    with open('../TopSense/data/orig_new.json') as f:
        orig_new = json.load(f)
    return word_sense2chdef_level, orig_new

word_sense2chdef_level, orig_new = load_data()

def find_target_word(sentence):
    pattern = r'\[[A-Za-z]*\]'
    result = re.search(pattern, sentence)
    targetword = result.group()[1:-1]
    return targetword

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

def show_results(targetword, topics_data, senses_data):
    st.subheader(f'Topis of "{targetword}"')
    for idx, data in enumerate(topics_data):
        supergroup = data['class']
        topic = data['text']
        score = data['score']
        st.markdown(f"{idx+1}. [{supergroup}] {topic}   {score}")
    
    st.subheader(f'Possible word sense of "{targetword}"')
    for idx, sense in enumerate(senses_data):
        en_def = sense['en_def']
        ch_def = sense['ch_def']
        level = sense['level']
        score = sense['score']
        guideword = sense['guideword']

        st.markdown(f"{idx+1}. {level} {guideword} {en_def} ----- {score}")
        st.markdown(ch_def)
        
sentence = st.text_area("Sentence here", help='place target word in the []',
                   value='A man was fishing on the opposite [bank].')
targetword = find_target_word(sentence)
clicked = st.button("Enter")

def main():
    tokens = tokenize_processor(sentence) 
    ranked_senses, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                      sentence, 
                                                                                      'noun',
                                                                                      targetword)
    topics_data, senses_data = gen_store_data(token_scores, ranked_senses, targetword)
    show_results(targetword, topics_data, senses_data)

if clicked:
    main()
