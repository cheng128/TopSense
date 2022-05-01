import json
import nltk
import pickle
import argparse
import wikipediaapi
import urllib.parse
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-t5-xl')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
wiki_wiki = wikipediaapi.Wikipedia(language='en')

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'n')
    return lemma

def load_data(version):  
    with open('../data/words2defs.json') as f:
        word2def = json.loads(f.read())
    
    with open('../data/sentence-t5-xl_sense_examples_embs.pickle', 'rb') as f:
        def_examples_embs = pickle.load(f)

    sents_filename = f'../data/wiki/{version}_wiki_href_word2sents.json'
    with open(sents_filename) as f:
        href2word_sents = json.loads(f.read())
    
    href_filename = f'../data/wiki/{version}_wiki_href2def.json'
    with open(href_filename) as f:
        href2def = json.loads(f.read())

    print('sents file:', sents_filename)
    print('href file:', href_filename)
    
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]
        
    def2id = defaultdict(list)
    for data in cambridge:
        if data['pos'] == 'noun':
            def2id[data['en_def']].append({'headword': data['headword'],
                                           'id': data['id']})
    
    return word2def, def2id, def_examples_embs, href2word_sents, href2def

def build_map(data):
    href_word2sents = defaultdict(list)
    for href, value in data.items():
        for pair in value:
            word, sent = pair
            href_word2sents[(href, word)].append(sent)
    return href_word2sents

def cal_similarity(first_sent, word, sense, def_examples_embs, sents):  
    # contains definition and examples sentences embeddings
    def_embs = def_examples_embs[(word, 'noun')][sense]['sense_emb']
    embeddings = model.encode([first_sent], convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings, def_embs)
    sense2sense = cosine_scores[0][0]

    final_score = sense2sense
    return final_score

def find_proper_sense(first_sent, word, word2def, def2id, def_examples_embs, sents): 
    # all senses of this word
    definitions = word2def[word]
    if len(definitions) == 1:
        sense = definitions[0]
        similarity = cal_similarity(first_sent, word, sense, def_examples_embs, sents)
        word_id = def2id[sense][0]['id']
        return word_id, similarity
    else:
        word_id = None
        max_similarity = float('-inf')
        proper_sense = ''

        for sense in definitions:
            similarity = cal_similarity(first_sent, word, sense, def_examples_embs, sents)
            if similarity > max_similarity:
                max_similarity = similarity
                proper_sense = sense
                for data in def2id[proper_sense]:
                    if data['headword'] == word:
                        word_id = data['id']
        return word_id, max_similarity

def process_sent(href2def, href_word2sents, word2def, def2id, def_examples_embs):
    href_word2id = {}
    # ('bass (fish)', 'bass')
    for href_word, sents in tqdm(href_word2sents.items()):
        href, word = href_word
        first_sent = href2def[href]
        word_id, score = find_proper_sense(first_sent, word, word2def, 
                                            def2id, def_examples_embs, sents)
        if word_id and score:
            href_word2id[(href, word)] = (word_id, float(score))
    return href_word2id

def gen_store_data(href_word2sents, href_word2id):
    word_id2sents = defaultdict(list)
    for href_word, sents in tqdm(href_word2sents.items()):
        if href_word in href_word2id:
            word_id, score = href_word2id[href_word]
            # put all word and sents that belongs to the word_id into the list
            # caution: one word id may be mapped by multiple hrefs
            word_id2sents[word_id].append([href_word[1], score, sents])

    # we only save the linked page data that has the highest score 
    new_id2sents = {}
    for key, value in word_id2sents.items():
        # if the score is the same, we'll save the one that has more sentences
        value = sorted(value, key=lambda x: len(x[-1]), reverse=True)
        scores = [line[1] for line in value]
        highest_score = max(scores)
        highest_idx = scores.index(highest_score)
        new_id2sents[key] = value[highest_idx]
    
    return new_id2sents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=str)
    args = parser.parse_args()
    version = args.v
    
    save_filename = f'../data/wiki/all_{version}_wiki_word_id2sents_highest.json'
    print("save file:", save_filename)

    word2def, def2id, def_examples_embs, href2word_sents, href2def = load_data(version)
    href_word2sents = build_map(href2word_sents)

    href_word2id = process_sent(href2def, href_word2sents, word2def, def2id, def_examples_embs)

    word_id2sents = gen_store_data(href_word2sents, href_word2id)

    with open(save_filename, 'w') as f:
        f.write(json.dumps(word_id2sents))
                
if __name__ == '__main__':
    main()