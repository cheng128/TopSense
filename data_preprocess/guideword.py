""" Try to add guide word into local related sentence to calculate similarity
"""

import json
import csv
import argparse
import queue as Q
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-roberta-large-v1')

def load_data(file_type, word):
    if file_type == 'test':
        filename = f'../data/remap_test_data/BRT_data_{word}_test.json'
    elif file_type == 'normal':
        filename = '../data/BRT_data.json'
        
    print('open_brt:', filename)
    with open(filename) as f:
        brt_data = json.loads(f.read())

    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]
    
    cam_map = defaultdict(list)
    for line in cambridge:
        cam_map[line['headword']].append(line)
    return brt_data, cambridge, cam_map

def cal_similarity(target, one_sense, guide_bool, way='avg'):
    sentences = [target['en_def']]
    if target['examples']:
        examples = [i['en'] for i in target['examples']]
        sentences.extend(examples)
    if target['guideword'] and guide_bool:
        sentences.append(target['guideword'][1:-1].lower())
    edge = len(sentences)
    sentences.extend(one_sense)

    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    pairs = []
    for i in range(edge):
        for j in range(edge, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    # decay
    if way == 'avg':
        similarity = sum([p['score'] for p in pairs]) / len(pairs)
    elif way == 'top3':
        similarity = sum([p['score'] for p in pairs][:3]) / 3
    elif way == 'top5':
        similarity = sum([p['score'] for p in pairs][:5]) / 5
        
    return similarity

def cal_related(definitions, category,  pos, guide_bool, way, related, sim_scores):
    for sense in definitions:
        if sense['pos'] == pos:
            sim_scores.append([sense, cal_similarity(sense, related, guide_bool, way)])
    return sim_scores

def cal_sim_scores(pos, category, local_related, global_related,
                   definitions, way, guide_bool):
    if not local_related and not global_related:
        local_related = [category]
    sim_scores = []
    # first considerate words that is not ambiguate or in the same group
    if local_related:
        sim_scores = cal_related(definitions,category,  pos, guide_bool, way, 
                                 local_related, sim_scores)
        
    elif global_related:
        sim_scores = cal_related(definitions,category,  pos, guide_bool, way, 
                                 global_related, sim_scores)
    sorted_sim_sense = sorted(sim_scores, key=lambda x:x[1], reverse=True)
        
    return sorted_sim_sense

def decide_filename(args):
    if args.f == 'test':
        filename = f'../data/jsonl_file/{args.g}_{args.r}_{args.c}_{args.w}.jsonl'
        write_file = f'../data/comparison/{args.g}_{args.r}_{args.c}_{args.w}.txt'
    elif args.f == 'normal':
        filename = f'../data/jsonl_file/{args.r}_{args.c}_{args.g}_brt_map_cam.jsonl'
        write_file = f'../data/comparison/{args.r}_{args.c}_{args.g}_all_test_guide.txt'
        
    print('file type: ', args.f)
    print('word:', args.w)
    print('filename:', filename)
    print('way: ', args.c, 'threshold: ', args.r)
    return filename, write_file

def gen_pos_map():
    pos_map = {'adj': 'adjective', 'adv':'adverb', 'noun': 'noun', 'verb': 'verb'}
    return pos_map

def write_file(filename, map_data):
    with open(filename, 'w') as f:
        for data in map_data:
            f.write(json.dumps(data))
            f.write('\n')

def initiate_related(super_group, category, guide_bool):
    if not guide_bool:
        return []
    else:
        return [super_group.lower(), category.lower()]
    
def gen_store_data(super_group, category, pos, word, group, word_id, sense, score):
    data = {'super_group': super_group,
            'category': category,
            'pos': pos,
            'brt_word': word,
            'group': group,
            'word_id': word_id,
            'en_def': sense['en_def'],
            'ch_def': sense['ch_def'],
            'score': score}
    return data

def update_related(word, score, threshold, local_related, global_related, sense):
    if score >= threshold:
        local_related.append(sense['en_def'])
        global_related.append(sense['en_def'])
        guideword = sense['guideword'].lower()
        if guideword:
            local_related.append(guideword[1:-1])
            global_related.append(guideword[1:-1])
    return local_related, global_related
    
def clear_q(threshold, q, map_data, word_def, super_group, category, pos, word, group, 
            global_related, local_related, way, guide_bool):
    while not q.empty():
        first_item = q.get()
        brt_word = first_item[1]
        definitions = word_def[brt_word]
        sorted_sim_sense = cal_sim_scores(pos, category, local_related,
                                          global_related, definitions,
                                          way, guide_bool)
        if sorted_sim_sense:
            most_sim_sense = sorted_sim_sense[0][0]
            score = float(sorted_sim_sense[0][1])
            word_id = most_sim_sense['id']
            store_data = gen_store_data(super_group, category, pos, brt_word, 
                                         group, word_id, most_sim_sense, score)
            map_data.append(store_data)
            local_related, global_related = update_related(brt_word, score, threshold, 
                                                           local_related, 
                                                           global_related, 
                                                           most_sim_sense)

        # update similarity score of all definitions in queue to rerank the queue
        for i in range(q.qsize()):
            first_item = q.get()
            brt_word = first_item[1]
            definitions = word_def[brt_word]
            sorted_sim_sense = cal_sim_scores(pos, category, local_related,
                                              global_related, definitions, 
                                              way, guide_bool)
            score = float(sorted_sim_sense[0][1])
            q.put([1 / score, brt_word])
    return map_data

def remap(args, filename, brt_data, cam_map):
    
    way, threshold = args.c, args.r
    filetype, test_word = args.f, args.w
    guide_bool = args.g
    
    pos_map = gen_pos_map()
    
    q = Q.PriorityQueue()
    
    map_data = []
    for super_group, categories in brt_data.items():
        print(super_group)
        write_file(filename, map_data)
        for category, pos_data in tqdm(categories.items()):
            for pos, groups in pos_data.items():
                if pos in ['cat_num', 'interjections']:
                    continue
                global_related = initiate_related(super_group, category, guide_bool)
                pos = pos_map[pos]
                for group, words in groups.items():
                    local_related = initiate_related(super_group, category, guide_bool)
                    word_def = {}
                    seen = []
                    for word in words:
                        definitions = cam_map.get(word, {})
                        # first deal with words that have only one sense
                        if len(definitions) == 1 and definitions[0]['pos'] == pos:
                            sense = definitions[0]
                            word_id = sense['id']
                            store_data = gen_store_data(super_group, category, pos, word,
                                                group, word_id, sense, 'one_sense')
                            map_data.append(store_data)
                            local_related, global_related = update_related(word, 1, 0, 
                                                                           local_related, 
                                                                           global_related, 
                                                                           sense)
                            
                            seen.append(word)
                        elif word.rsplit() and word.rsplit()[-1].isdigit():
                            word_cat = ' '.join(word.rsplit()[:-1])
                            local_related.append(word_cat.lower())
                        
                    for word in words:
                        if word not in seen:
                            definitions = cam_map.get(word, {})
                            if definitions:
                                sorted_sim_sense = cal_sim_scores(pos, category, local_related,
                                                                  global_related, definitions, 
                                                                  way, guide_bool)
                                if sorted_sim_sense:
                                    # highest similarity score in all senses
                                    score = float(sorted_sim_sense[0][1])
                                    q.put([1 / score, word])
                                    word_def[word] = definitions
                                    
                    map_data = clear_q(threshold, q, map_data, word_def, super_group, category, 
                                       pos, word, group, global_related, local_related, way, 
                                       guide_bool)
        write_file(filename, map_data)
        
def main():
    parser = argparse.ArgumentParser()
    # the way to calculate similarity, top3, top5 or avg
    parser.add_argument('-c', type=str)
    # the threshold of add a sentence into local, global related
    parser.add_argument('-r', type=float)
    # file type = 'test' or 'normal'
    parser.add_argument('-f', type=str)
    # if it is a test, then we can use a word to test the result
    parser.add_argument('-w', type=str, default='all')
    # add guideword into local, global related
    parser.add_argument('-g', type=bool)
    args = parser.parse_args()
    
    filename, write_file = decide_filename(args)
    brt_data, _, cam_map = load_data(args.f, args.w)
    
    remap(args, filename, brt_data, cam_map)
    
    print('comparison: ', write_file)

    with open(filename) as f:
        remap_data = [json.loads(line) for line in f.readlines()]

    with open('../data/BRT.super.disamb.tsv') as f:
        rows = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in rows:
             for data in remap_data:
                if row[0] == data['super_group'] and \
                row[2] == data['category'] and \
                row[3] == 'noun' and row[3] == data['pos'] and \
                row[4] == data['group'] and \
                row[5] == data['brt_word']:
                    ans = ''
                    for i in row[6]:
                        if i == '(':
                            break
                        ans += i
                    word_id = ans.replace('-', '.').split('.')
                    word_id[-1] = word_id[-1].zfill(2)
                    word_id = '.'.join(word_id)
                    if word_id != data['word_id']:
                        with open(write_file, 'a') as f:
                            f.write(data['category'] + ' ' + row[4] +'\n')
                            f.write(word_id + ' ' + row[-2]+'\n')
                            f.write(data['word_id'] + ' ' + data['en_def'] + '\n')
    
    
if __name__ == '__main__':
    main()