import json
from tqdm import tqdm
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

def load_data():
    filename = 'refactor_True_0_top3_map_cross_reference.jsonl'
    print(filename)
    with open(f'../data/jsonl_file/{filename}') as f:
        remap_data = [json.loads(line) for line in f.readlines()]

    word_category = defaultdict(dict)
    for line in remap_data:
        if line['score'] == 1:
            word = line['brt_word']
            category = line['category']
            if 'topics' not in word_category[word]:
                word_category[word][line['pos']] =[category]
                word_category[word]['sentences'] = []
            else:   
                word_category[word][line['pos']].append(category)
    
    wiki_data = []
    with open('../data/wiki/pure_sentences.txt') as f:
        for line in f.readlines():
            wiki_data.append(line)

    return word_category, wiki_data   

def main():
    word_category, wiki_data = load_data()

    for word in tqdm(word_category): 
        for sent in wiki_data:
            if ' ' + word + ' ' in sent:
                word_category[word]['sentences'].append(sent)

    delete_list = []
    for word, value in word_category.items():
        if 'sentences' not in value:
            delete_list.append(word)

    for word in delete_list:
        del word_category[word]

    with open('../data/monosemous_sents.json', 'w') as f:
        f.write(json.dumps(word_category))
    
if __name__ == '__main__':
    main()