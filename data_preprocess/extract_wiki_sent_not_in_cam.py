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

    category_words = defaultdict(list)
    for line in remap_data:
        category_words[line['super_group'], line['category'], line['pos'], line['group']].append(line['brt_word'])

    with open('../data/BRT_data.json') as f:
        brt_data = json.load(f)
    
    # print('parse Wikipedia sentences')
    # with open('../data/wiki/simple_en_wiki_link.txt') as f:
    #     wiki_data = []
    #     for line in tqdm(f.readlines()):
    #         text = json.loads(line)['text']
    #         soup = BeautifulSoup(text, 'lxml')
    #         pure_text = soup.text
    #         wiki_data.extend(sent_tokenize(pure_text))

    wiki_data = []
    with open('../data/wiki/pure_sentences.txt') as f:
        for line in f.readlines():
            wiki_data.append(line)

    return remap_data, category_words, wiki_data, brt_data

def build_map(category_words, brt_data):

    words_list = defaultdict(dict)
    for supergroup, value in tqdm(brt_data.items()):
        for category, pos_data in value.items():
            for pos, groups in pos_data.items():
                if pos in ['cat_num', 'interjection']:
                    continue
                for group, words in groups.items():
                    for word in words:
                        if word not in category_words[supergroup, category, pos, group]:
                            if pos in words_list[word]:
                                words_list[word][pos].append(category)
                            else:
                                words_list[word][pos] = [category]
                                words_list[word]['sentences'] = [] 

    return words_list    

def main():
    remap_data, category_words, wiki_data, brt_data = load_data()
    words_list = build_map(category_words, brt_data)

    for word in tqdm(words_list): 
        for sent in wiki_data:
            if ' ' + word + ' ' in sent: 
                words_list[word]['sentences'].append(sent)

    delete_list = []
    for word, value in words_list.items():
        if 'sentences' not in value:
            delete_list.append(word)

    for word in delete_list:
        del words_list[word]

    with open('../data/new_not_in_cam_sents.json', 'w') as f:
        f.write(json.dumps(words_list))
    
if __name__ == '__main__':
    main()