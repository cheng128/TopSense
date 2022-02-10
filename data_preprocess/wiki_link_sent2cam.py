# remove disambiguate pages
# left sense to embed only

import json
import nltk
import pickle
import wikipediaapi
import urllib.parse
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import pdb

model = SentenceTransformer('all-roberta-large-v1')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
wiki_wiki = wikipediaapi.Wikipedia(language='en')

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def load_data():  
    with open('../data/words2defs.json') as f:
        word2def = json.loads(f.read())
    
    with open('../data/cam_def2id_word.json') as f:
        def2id = json.loads(f.read())
    
    with open('../data/definition_emb.pickle', 'rb') as f:
        def_emb_dict = pickle.load(f)
        
    with open('../data/wiki/wiki_href_word2sents.json') as f:
        href2word_sents = json.loads(f.read())
        
    with open('../data/wiki/wiki_href2def.json') as f:
        href2def = json.loads(f.read())
    
    return word2def, def2id, def_emb_dict, href2word_sents, href2def

def build_map(data):
    href_word2sents = defaultdict(list)
    for href, value in data.items():
        for pair in value:
            word, sent = pair
            href_word2sents[(href, word)].append(sent)
    return href_word2sents


def cal_similarity(first_sent, sense, def_emb_dict):  
    # contains definition and examples sentences and guide word embs
    def_embs = def_emb_dict[sense]
    embeddings = model.encode([first_sent], convert_to_tensor=True)

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embeddings, def_embs)

    #Find the pairs with the highest cosine similarity scores
    similarity = 0
    for j in range(1, len(def_embs)):
#         pairs.append({'index': [0, j], 'score': cosine_scores[0][j]})
        similarity += cosine_scores[0][j]
    
    avg = similarity / len(def_embs)
    pdb.set_trace()
    return avg

def find_proper_sense(first_sent, word, word2def, def2id, def_emb_dict):
    
    # all senses of this word
    definitions = word2def[word]
    if len(definitions) == 1:
        word_id = def2id[definitions[0]]['id']
        return word_id, 1
    else:
        word_id = None
        max_similarity = float('-inf')
        proper_sense = ''

        for sense in definitions:
            similarity = cal_similarity(first_sent, sense, def_emb_dict)
            if similarity > max_similarity:
                max_similarity = similarity
                proper_sense = sense
                word_id = def2id[proper_sense]['id']
        return word_id, max_similarity

def process_sent(href2def, href_word2sents, word2def, def2id, def_emb_dict):
    href_word2id = {}
    # ('bass (fish)', 'bass')
    for href_word in tqdm(href_word2sents):
        href, word = href_word
        first_sent = href2def[href]
        word_id, score = find_proper_sense(first_sent, word, word2def, def2id, def_emb_dict)
        if word_id and score:
            href_word2id[(href, word)] = (word_id, float(score))
    return href_word2id


def main():
    word2def, def2id, def_emb_dict, href2word_sents, href2def = load_data()
    href_word2sents = build_map(href2word_sents)

    href_word2id = process_sent(href2def, href_word2sents, word2def, def2id, def_emb_dict)

    word_id2sents = defaultdict(dict)
    for href_word, sents in tqdm(href_word2sents.items()):
        if href_word in href_word2id:
            word_id, score = href_word2id[href_word]
            word_id2sents[word_id] = [href_word[1], score, sents]

#     with open('../data/wiki/wiki_word_id2sents.json', 'w') as f:
#         f.write(json.dumps(word_id2sents))
                    

if __name__ == '__main__':
    main()