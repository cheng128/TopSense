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
    with open('../data/words2defs.json') as f:
        word2def = json.loads(f.read())
    
    with open('../data/remap.word_id.topics.examples.json') as f:
        word_id2topics = json.loads(f.read())
        
    target_words = set([value['headword'] 
                        for key, value in word_id2topics.items()
                        if value['topics']])
    
    return word2def, target_words

def build_map(sent, page_def, href_map, word2def, target_words, wiki_wiki):
    soup = BeautifulSoup(sent, 'lxml')

    # deal with each linked word in sentence
    for tag in soup.find_all('a'):
        word = tag.string
        # transform url string into normal text
        if word and word in word2def and word in target_words: 
            href = urllib.parse.unquote(tag['href'])
            if href not in page_def:
                is_disambiguate = False
                page = wiki_wiki.page(href)
                try:
                    if page.exists():
                        for category in page.categories:
                            if 'disambiguation pages' in category.lower():
                                is_disambiguate = True
                                print(page.title)
                                break
                        # use first sentence as definition of this page
                        first_sent = tokenizer.tokenize(page.summary)[0]
                        if first_sent and not is_disambiguate:
                            page_def[href] = first_sent
                except:
                    continue
            else:
                href_map[href].append([word, soup.text])
            
    return page_def, href_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=str)
    args = parser.parse_args()
    
    version = args.v
    print(f'{version} wiki')
    
    href2def = f'../data/wiki/{version}_wiki_href2def.json'
    print("href2def:", href2def)

    word2sents = f'../data/wiki/{version}_wiki_href_word2sents.json'
    print("word2sents:", word2sents)

    wiki_wiki = wikipediaapi.Wikipedia(language=version)
    
    word2def, target_words = load_data()
    word_id2wiki_sent = defaultdict(list)
    
    page_def = {}
    href_map = defaultdict(list)
    with open(f'../data/wiki/{version}_en_wiki_link.txt') as f:
        count = 0
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            article = data['text']

            sentences = tokenizer.tokenize(article)
            
            for sent in sentences:
                page_def, href_map = build_map(sent, page_def, href_map, 
                                                word2def, target_words, wiki_wiki)
            count += 1
            if count % 10000 == 0:
                with open(href2def, 'w') as f:
                    f.write(json.dumps(page_def))
                    
                with open(word2sents, 'w') as f:
                    f.write(json.dumps(href_map)) 

    with open(href2def, 'w') as f:
        f.write(json.dumps(page_def))
        
    with open(word2sents, 'w') as f:
        f.write(json.dumps(href_map))

if __name__ == '__main__':
    main()