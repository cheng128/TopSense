import json
import string
import spacy
import urllib
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
from html import unescape

nlp = spacy.load("en_core_web_sm")

BOOK_API = 'https://www.googleapis.com/books/v1/volumes?'
NGRAM_API = 'https://linggle.com/api/ngram/'
EXAMPLE_API = 'https://linggle.com/api/example/'

def encode(query):
    return urllib.parse.quote(query)

def get_ngram(query):
    '''Please refer to https://linggle.com/help/ for the query format'''

    result = requests.get(NGRAM_API + encode(query))
    if not result.ok:
        print(f'get ngrams request failed: {query}')
        return
    return result.json()

def get_example(ngram):
    result = requests.post(EXAMPLE_API, json={
                 'ngram': ngram
             })
    if not result.ok:
        print(f'example request failed: {ngram}')
        return
    
    result = result.json()['examples']
    
    book_result = requests.get(BOOK_API + f'q="{ngram}"&maxResults=40')
    if not book_result.ok:
        print(f'book request failed: {ngram}')
        return result
    
    try:
        for item in book_result.json()['items']:
            try:
                result.append(item['searchInfo']['textSnippet'])
            except:
                pass
    except:
        return result
    
    clean_sents = []
    for s in result:
        s = BeautifulSoup(unescape(s), 'lxml').text
        s = s.replace(u'\xa0', u' ')
        clean_sents.append(s)
    
    return clean_sents

def find_collo(words_list):
    count = 2
    temp = []
    for word in words_list:
        if count == 0:
            break
        if not word.is_stop and word.text not in string.punctuation\
        and not word.text.isdigit():
            temp.append(word.lemma_)
            count -= 1
    return temp

def fetch_query(word, sent, not_found=False):
    doc = nlp(sent)
    anchor = -1
    for idx, token in enumerate(doc):
        if token.text.lower() == word.lower() or token.lemma_.lower() == word.lower():
            anchor = idx
    if anchor == -1:
        with open('../data/not_found_examples.txt', 'a') as f:
            f.write(f'word: {word}, sent: {sent}')
        return ''
    
    query = ["{"]

    first_half = find_collo(list(doc)[anchor-1:anchor-5:-1])
    
    if not not_found:
        first_half = find_collo(list(doc)[anchor-1:anchor-3:-1])
    if first_half:
        query.extend(first_half[::-1])

    last_half = find_collo(list(doc)[anchor+1:anchor+5])
    
    if not not_found:
        last_half = find_collo(list(doc)[anchor+1:anchor+3])
    if last_half:
        query.extend(last_half)

    query.extend([word, '_', '?_', '?_', '}'])
    query = ' '.join(query)
    
    return query
    
def main():

    with open('../data/remap.word_id.topics.examples.json') as f:
        word_id2topics = json.loads(f.read())
    
    with open('../data/remap.augment.word_id.topics.examples.json') as f:
        already_fetched = json.loads(f.read())

    for word, value in tqdm(word_id2topics.items()):
        if word not in already_fetched:
            examples = []
            headword = value['headword']
            if value['examples']:
                for sent in value['examples']:
                    query = fetch_query(value['headword'], sent)
                    ngrams = get_ngram(query)
                    # try again with another query
                    if not ngrams or not ngrams['ngrams']:
                        query = fetch_query(headword, sent, True)
                        ngrams = get_ngram(query)
                    # still can't find result
                    if not ngrams or not ngrams['ngrams']:
                        with open('../data/remap_not_found_examples.txt', 'a') as f:
                            f.write(f'word: {headword}, query: {query}, sent: {sent}\n')
                        pass
                    else:
                        first_ngram = ngrams['ngrams'][0][0]
                        examples = get_example(first_ngram)
            already_fetched[word] = {'pos':value['pos'],
                                  'headword': headword,
                                  'topics': value['topics'],
                                  'examples': value['examples'] + examples}

        with open('../data/remap.augment.word_id.topics.examples.json', 'w') as f:
            f.write(json.dumps(already_fetched))
    
if __name__ == '__main__':
    main()