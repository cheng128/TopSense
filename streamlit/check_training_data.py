import json
import streamlit as st


st.title('TopSense Training Data')

def load_data():
    with open('../TopSense/data/BRT_data.json') as f:
        brt_data = json.load(f)
    
    with open('../TopSense/data/0.6.word_id.topics.examples.json') as f:
        word2topics = json.load(f)
        
    with open('../TopSense/data/word_id2sense.json') as f:
        word_id2sense = json.load(f)
        
    return brt_data, word2topics, word_id2sense

brt_data, word2topics, word_id2sense = load_data()

def show_category_words(category_name, pos_tag, targetword=''):
    for supergroup, categories in brt_data.items():
        for category, pos_data in categories.items():
            for pos, word_data in pos_data.items():
                if pos == 'cat_num':
                    continue
                if category_name == category and pos == pos_tag:
                    for group, word_sets in word_data.items():
                        st.markdown(group)
                        if targetword in word_sets:
                            st.markdown(f"[FOUND TARGETWORD]: {targetword}")
                        elif targetword and targetword not in word_sets:
                            st.markdown("[TARGETWORD NOT FOUND]")
                        for word in word_sets:
                            st.markdown(word)
                        st.markdown('-' * 30)
                        
def find_training_data(targetword, pos):
    found = False
    for word_id, data in word2topics.items():
        if data['headword'] == targetword and data['pos'] == pos:
            found = True
            st.markdown("[ SENSE ]")
            st.markdown(word_id2sense[word_id]['en_def'])
            st.markdown(word_id2sense[word_id]['ch_def'])
            st.markdown('')
            st.markdown('[ EXAMPLES ]')
            for example in data['examples']:
                st.markdown(example)
            st.markdown('')
            st.markdown("[ TOPICS]")
            for topic in set(data['topics']):
                st.markdown('[' + topic + ']')
            st.markdown('-' * 30)
    if not found:
        st.markdown(f"There's no training data for {targetword}, {pos}")
    

targetword, category_name = "", ""

targetword = st.text_input("Targetword (e.g., represent)")
category_name = st.text_input("Category Name (e.g., List)")
pos_choice = st.selectbox("POS tag", ["noun", "verb"])
clicked = st.button("Submit")

def main():
    if targetword and category_name:
        show_category_words(category_name, pos_choice, targetword)
    elif targetword:
        find_training_data(targetword, pos_choice)
    elif category_name:
        show_category_words(category_name, pos_choice)
    return 
    
if clicked:
    main()
