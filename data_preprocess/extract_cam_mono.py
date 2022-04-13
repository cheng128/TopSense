import json
from tqdm import tqdm
from collections import defaultdict

def load_data():
    filename = 'refactor_True_0_top3_map_cross_reference.jsonl'
    print(filename)
    with open(f'../data/jsonl_file/{filename}') as f:
        remap_data = [json.loads(line) for line in f.readlines()]
        
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]

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
                
    cambridge_sents = []
    for line in cambridge:
        examples = [data['en'] for data in line['examples']]
        cambridge_sents.extend(examples)
    
    return word_category, cambridge_sents


def main():
    word_category, cambridge_sents = load_data()

    for word in tqdm(word_category): 
        for sent in cambridge_sents:
            if word in sent:
                word_category[word]['sentences'].append(sent)

    delete_list = []
    for word, value in word_category.items():
        if not value['sentences']:
            delete_list.append(word)

    for word in delete_list:
        del word_category[word]

    with open('../data/cam_monosemous_sents.json', 'w') as f:
        f.write(json.dumps(word_category))
    
if __name__ == '__main__':
    main()