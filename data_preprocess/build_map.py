import json
import nltk
import wikipediaapi
import urllib.parse
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
wiki_wiki = wikipediaapi.Wikipedia(language='en')


def load_data():
    with open('../data/words2defs.json') as f:
        word2def = json.loads(f.read())
    
    return word2def

def build_map(sent, page_def, href_map, word2def):
    soup = BeautifulSoup(sent, 'lxml')

    # deal with each linked word in sentence
    for tag in soup.find_all('a'):
        word = tag.string
        # transform url string into normal text
        if word and word in word2def:
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
    word2def = load_data()
    word_id2wiki_sent = defaultdict(list)
    
    page_def = {}
    href_map = defaultdict(list)
    with open('../data/wiki/simple_en_wiki_link.txt') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            article = data['text']

            sentences = tokenizer.tokenize(article)
            
            for sent in sentences:
                page_def, href_map = build_map(sent, page_def, href_map, word2def)

    with open('../data/wiki/wiki_href2def.json', 'w') as f:
        f.write(json.dumps(page_def))
        
    with open('../data/wiki/wiki_href_word2sents.json', 'w') as f:
        f.write(json.dumps(href_map))

if __name__ == '__main__':
    main()