"""
This file is used to fetch the first sent of a Wikipedia page and 
save all sentences that link to the page.
"""
import json
import nltk
import argparse
import wikipediaapi
import urllib.parse
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def load_data():
    
    with open('../data/0.0.word_id.topics.examples.json') as f:
        word_id2topics = json.load(f)
        
    target_words = set([value['headword'] 
                        for key, value in word_id2topics.items()
                        if value['topics'] and value['pos'] == 'noun'])
    
    return target_words

def build_map(sent, page_def, href_map, target_words, wiki_wiki):
    soup = BeautifulSoup(sent, 'lxml')

    # deal with each linked word in sentence
    for tag in soup.find_all('a'):
        word = tag.string
        # check if it's a noun, and have topics
        if word in target_words: 
            href = urllib.parse.unquote(tag['href'])
            # haven't been fatched yet
            if href not in page_def:
                is_disambiguate = False
                page = wiki_wiki.page(href)
                try:
                    if page.exists():
                        # check if it's a disambiguation page
                        for category in page.categories:
                            if 'disambiguation pages' in category.lower():
                                is_disambiguate = True
                                break
                        # use first sentence as definition of this page
                        first_sent = tokenizer.tokenize(page.summary)[0]
                        if first_sent and not is_disambiguate:
                            page_def[href] = first_sent
                except:
                    continue
            else:
                # save the linked word in sentence to replace it with [MASK] in the next step
                if [word, soup.text] not in href_map[href]:
                    href_map[href].append([word, soup.text])
            
    return page_def, href_map

def load_processed_file(href2def, word2sents):
    
    with open(href2def) as f:
        page_def = json.load(f)
    
    with open(word2sents) as f:
        href_map = defaultdict(list, json.load(f))
        
    return page_def, href_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=str)
    args = parser.parse_args()
    
    version = args.v
    print(f'{version} wiki')
    
    href2def = f'../data/wiki/{version}_wiki_href2def.json'
    word2sents = f'../data/wiki/{version}_wiki_href_word2sents.json'

    print("href2def:", href2def)
    print("word2sents:", word2sents)

    wiki_wiki = wikipediaapi.Wikipedia(language=version)
    
    target_words = load_data()
    word_id2wiki_sent = defaultdict(list)

#     in case that the program will be killed when we run English Wikipedia data
#     page_def, href_map = load_processed_file(href2def, word2sents)

    page_def = {}
    href_map = defaultdict(list)
    with open(f'../data/wiki/{version}_en_wiki_link.txt') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            article = data['text']

            sentences = tokenizer.tokenize(article)
            
            for sent in sentences:
                page_def, href_map = build_map(sent, page_def, href_map, 
                                                target_words, wiki_wiki)
    
    with open(href2def, 'w') as f:
        f.write(json.dumps(page_def))
    
    with open(word2sents, 'w') as f:
        f.write(json.dumps(href_map))

if __name__ == '__main__':
    main()