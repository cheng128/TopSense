import json
import nltk
import spacy
import requests
from tqdm import tqdm
from collections import Counter, defaultdict
from nltk.corpus import stopwords

spacy_model = spacy.load("en_core_web_sm")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

ngram_api = 'https://linggle.com/ngram/'
examples_api = 'http://whisky.nlplab.cc:9489/example/coca'
headers = {'user-agent': 'Mozilla/5.0'}

def load_data():
    with open('../data/boostrap_word_id.topics.examples.json') as f:
        word_examples = json.load(f)
        
    return word_examples

def is_valid_pos(token):
    return token.pos_ in ['NOUN']

def is_collo_valid_pos(token):
    return token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']

def is_not_stopword(token):
    return token.lemma_.lower() not in spacy_stopwords

def gen_query(sent, sentence_data, local_candidate, targetword):
    sentence_data[sent]['candidate'] = local_candidate
    if targetword in local_candidate:
        target_idx = local_candidate.index(targetword)
        if target_idx != 0:
            preword = local_candidate[target_idx - 1]
        else:
            preword = ''

        if target_idx != (len(local_candidate)-1):
            postword = local_candidate[target_idx + 1]
        else:
            postword = ''

        if preword and postword:
            sentence_data[sent]['query'] = '{' + '~' + preword + '/~' + postword + ' ' + targetword + ' *}'
        elif preword:
            sentence_data[sent]['query'] = '{' + ' ~' + preword + ' ' + targetword + ' *}'
        elif postword:
            sentence_data[sent]['query'] = '{' + ' ~' + postword + ' ' + targetword + ' *}'
        else:
            sentence_data[sent]['query'] = ''
    else:
        sentence_data[sent]['query'] = ''
    return sentence_data

def fetch_ngrams(sentence_data, sent, value):
    query = value['query']
    if query:
        response = requests.get(ngram_api + query, headers=headers)
        try:
            sentence_data[sent]['ngrams'] = response.json()['ngrams']
        except:
            print('ngrams API')
            print(sent)
            print(query)
            print(response)
            print('-' * 30)
            sentence_data[sent]['ngrams'] = []
    else:
        sentence_data[sent]['ngrams'] = []
    return sentence_data

def find_collocations(collocations, sentence):
    for token in spacy_model(sentence):
        if is_not_stopword(token) and is_collo_valid_pos(token):
            collocations.append(token.lemma_.lower())
    return collocations

def find_ngram_collocations(sentence_data, sent, value, targetword):
    ngrams = value['ngrams']
    if ngrams:
        collocations = []
        for ngram, times in ngrams:
            collocations = find_collocations(collocations, ngram)
        collocations = [i for i in sorted(Counter(collocations).items(), key=lambda x:x[1], reverse=True)
                       if i[0] != targetword]
        sentence_data[sent]['ngrams_collocation'] = collocations
    else:
        sentence_data[sent]['ngrams_collocation'] = []
    return sentence_data

def fetch_examples(sentence_data, sent, value):
    sentence_data[sent]['examples'] = {}
    ngrams = value['ngrams']
    for ngram, times in ngrams:
        query = {'ngram':ngram}
        response = requests.post(examples_api, json=query)
        try:
            sentence_data[sent]['examples'][ngram] = response.json()['examples']
        except:
            print('examples API')
            print(ngram)
            print(response.status_code)
            print('-' * 30)
            sentence_data[sent]['examples'][ngram] = []
    return sentence_data

def find_examples_collocations(sentence_data, sent, value, targetword):
    examples = value['examples']
    collocations = []
    for query, response in examples.items():
        if response:
            for text_data in response:
                example_sentence = text_data['text']
                collocations = find_collocations(collocations, example_sentence)
    collocations = [i for i in sorted(Counter(collocations).items(), key=lambda x:x[1], reverse=True)
                   if i[0] != targetword]
    sentence_data[sent]['examples_collocation'] = collocations
    return sentence_data
        
def main():
    word_examples = load_data()
    
    for word_id, word_data in tqdm(word_examples.items()):
        if word_data['pos'] == 'verb':
            targetword = word_data['headword']
            examples = word_data['examples']
            
            if type(examples) == dict or type(examples) == defaultdict:
                sentence_data = examples
                for sent, value in sentence_data.items():
                    if 'query' not in value:
                        local_candidate = []
                        for token in spacy_model(sent):
                            if (is_not_stopword(token) and is_valid_pos(token)) or token.text == targetword:
                                local_candidate.append(token.lemma_)
                        sentence_data = gen_query(sent, sentence_data, local_candidate, targetword)
                    # we have query, then we can check other items
                    else:
                        if 'ngrams' not in value or not value['ngrams']:
                            sentence_data = fetch_ngrams(sentence_data, sent, value)

                        if 'ngrams_collocation' not in value or not value['ngrams_collocation']:
                            sentence_data = find_ngram_collocations(sentence_data, sent, value, targetword)

                        # maybe we failed to fetch ngram examples ar the first time
                        if 'examples' not in value or not value['examples']: 
                            sentence_data = fetch_examples(sentence_data, sent, value)

                        if 'examples_collocation' not in value or not value['examples_collocation']:
                            sentence_data = find_examples_collocations(sentence_data, sent, value, targetword)
            
            elif type(examples) == list:
                sentence_data = defaultdict(dict)
                for sent in examples:
                    local_candidate = []
                    for token in spacy_model(sent):
                        if (is_not_stopword(token) and is_valid_pos(token)) or token.text == targetword:
                            local_candidate.append(token.lemma_)
                    if targetword in local_candidate:
                        sentence_data = gen_query(sent, sentence_data, local_candidate, targetword)
                    else:
                        continue

                    sentence_data = fetch_ngrams(sentence_data, sent, value)
                    sentence_data = find_ngram_collocations(sentence_data, sent, value, targetword)

                    sentence_data = fetch_examples(sentence_data, sent, value)
                    sentence_data = find_examples_collocations(sentence_data, sent, value, targetword)

            word_examples[word_id]['examples'] = sentence_data
                
        with open('../data/add_boostrap_word_id.topics.examples.json', 'w') as f:
            f.write(json.dumps(word_examples))
    
    return 

if __name__ == '__main__':
    main()