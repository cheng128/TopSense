import json
import argparse
from collections import defaultdict

def load_data():
    with open('../data/jsonl_file/refactor_True_0_top3_map_cross_reference.jsonl') as f:
        brt = [json.loads(line) for line in f.readlines()]
    
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]
    
    id2examples = {line['id']: [p['en'] for p in line['examples']] 
                   for line in cambridge}
    
    with open('../data/category_num.json') as f:
        cat_num = json.load(f)
    
    return brt, id2examples, cat_num

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=float, default=0)
    args = parser.parse_args()
    threshold = args.t
    print('threshold:', threshold)
    
    brt, id2examples, cat_num = load_data()
    
    id2topics = defaultdict(list)
    
    for data in brt:
        word_id = data['word_id']
        category = data['category']
        score = data['score']
        category_num = cat_num[category]
        if score >= threshold:
            id2topics[word_id].append(category_num + ' ' + category)
    
    id2topics_examples = defaultdict(dict)
    
    for data in brt:
        word_id = data['word_id']
        examples = id2examples[word_id]
        id2topics_examples[word_id]['pos'] = data['pos']
        id2topics_examples[word_id]['headword'] = data['brt_word']
        id2topics_examples[word_id]['examples'] = examples
        id2topics_examples[word_id]['topics'] = id2topics[word_id]
    
    with open(f'../data/{threshold}_remap.word_id.topics.examples.json', 'w') as f:
        f.write(json.dumps(id2topics_examples))
    
if __name__ == '__main__':
    main()