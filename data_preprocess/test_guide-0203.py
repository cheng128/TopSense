# Change queue to list

""" Try to add guide word into local related sentence to calculate similarity
"""

import json
import csv
import argparse
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
    
    return brt_data, cam_map

def cal_similarity(target, one_sense, way='avg'):
    sentences = [target['en_def']]
    if target['examples']:
        examples = [i['en'] for i in target['examples']]
        sentences.extend(examples)
    if target['guideword']:
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
        
    return similarity

def cal_sim_scores(pos, category, local_related, global_related, definitions, way, guide):
    if not local_related and not global_related:
        local_related = [category]
    sim_scores = []
    # first considerate words that is not ambiguate or in the same group
    if local_related:
        for sense in definitions:
            if sense['pos'] == pos:
                if sense['guideword'] and guide:
                    embeddings = model.encode([category, sense['guideword'][1:-1].lower()])
                    cosine_scores = util.cos_sim(embeddings, embeddings)
                    similarity = cosine_scores[0][1]
                    sim_scores.append([sense, 
                                       (cal_similarity(sense, local_related, way)+similarity)/2])
                else:
                    sim_scores.append([sense, cal_similarity(sense, local_related, way)])
    elif global_related:
        for sense in definitions:
            if sense['pos'] == pos:
                if sense['guideword'] and guide:
                    embeddings = model.encode([category, sense['guideword'][1:-1].lower()])
                    cosine_scores = util.cos_sim(embeddings, embeddings)
                    similarity = cosine_scores[0][1]
                    sim_scores.append([sense, 
                                       (cal_similarity(sense, global_related, way)+similarity)/2])
                else:
                    sim_scores.append([sense, cal_similarity(sense, global_related, way)])
    sorted_sim_sense = sorted(sim_scores, key=lambda x:x[1], reverse=True)
        
    return sorted_sim_sense

def init_related(guide_bool, super_group, category):
    related = []
    if guide_bool:
        related = [super_group.lower(), category.lower()]
    return related
    
def find_max(queue):
    max_item = queue[0]
    temp = []
    for item in queue[1:]:
        if item[0] > max_item[0]:
            temp.append(max_item)
            max_item = item
        else:
            temp.append(item)
    return max_item, temp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str)
    parser.add_argument('-r', type=float)
    parser.add_argument('-f', type=str)
    parser.add_argument('-w', type=str, default='all')
    parser.add_argument('-g', type=bool)
    args = parser.parse_args()
    word = args.w
    way = args.c
    threshold = args.r
    file_type = args.f
    guide = args.g
    
    if file_type == 'test':
        filename = f'../data/jsonl_file/{guide}_{threshold}_{way}_{word}_lower_test.jsonl'
        write_file = f'../data/comparison/{guide}_{threshold}_{way}_{word}_lower_test.txt'
    elif file_type == 'normal':
        filename = f'../data/jsonl_file/{guide}_{threshold}_{way}_brt_map_cam.jsonl'
        write_file = f'../data/comparison/{guide}_{threshold}_{way}_all_test_guide.txt'
    
    print('way: ', way)
    print('threshold: ', threshold)
    print('file type: ', file_type)
    print('word:', word)
    print('filename:', filename)
    
    
    brt_data, cam_map = load_data(file_type, word)
    pos_map = {'adj': 'adjective', 'adv':'adverb', 'noun': 'noun', 'verb': 'verb'}

    map_data = []
    for super_group, categories in brt_data.items():
        print(super_group)
        with open(filename, 'w') as f:
            for data in map_data:
                f.write(json.dumps(data))
                f.write('\n')
        for category, pos_data in tqdm(categories.items()):
            for pos, groups in pos_data.items():
                global_related = init_related(guide, super_group, category)
                if pos in ['cat_num', 'interjections']:
                    continue
                pos = pos_map[pos]
                for group, words in groups.items():
                    local_related = init_related(guide, super_group, category)
                    queue = []
                    seen = []
                    for word in words:
                        definitions = cam_map.get(word, {})
                        # only one possible sense, then it can be a reference of other words
                        if len(definitions) == 1 and definitions[0]['pos'] == pos:
                                seen.append(word)
                                sense = definitions[0]
                                map_data.append({'super_group': super_group,
                                                 'category': category,
                                                 'pos': pos,
                                                 'brt_word': word,
                                                 'group': group,
                                                 'word_id': sense['id'],
                                                 'en_def': sense['en_def'],
                                                 'ch_def': sense['ch_def'],
                                                 'score': 1})
                                local_related.append(sense['en_def'])
                                global_related.append(sense['en_def'])
                                guideword = sense['guideword'].lower()
                                if guideword and guide:
                                    local_related.append(guideword[1:-1])
                                    global_related.append(guideword[1:-1])
                                local_related = list(set(local_related))
                                global_related = list(set(global_related))
                    # if there are multi senses
                    for word in words:
                        if word not in seen:
                            definitions = cam_map.get(word, {})
                            if definitions:
                                sorted_sim_sense = cal_sim_scores(pos, category, local_related,
                                                                  global_related, definitions, 
                                                                  way, guide)
                                if sorted_sim_sense:
                                    # highest similarity score in all senses
                                    score = float(sorted_sim_sense[0][1])
                                    queue.append([score, word])  
                    while queue:
                        first_item, queue = find_max(queue)
                        brt_word = first_item[1]
                        definitions = cam_map[brt_word]
                        sorted_sim_sense = cal_sim_scores(pos, category, local_related,
                                                          global_related, definitions,
                                                          way, guide)
                        if sorted_sim_sense:
                            most_sim_sense = sorted_sim_sense[0][0]
                            score = float(sorted_sim_sense[0][1])
                            map_data.append({'super_group': super_group,
                                             'category': category,
                                             'pos': pos,
                                             'brt_word': brt_word,
                                             'group': group,
                                             'word_id': most_sim_sense['id'],
                                             'en_def': most_sim_sense['en_def'],
                                             'ch_def': most_sim_sense['ch_def'],
                                             'score': score})
                            if score >= threshold:
                                local_related.append(most_sim_sense['en_def'])
                                global_related.append(most_sim_sense['en_def'])
                                guideword = most_sim_sense['guideword'].lower()
                                if guideword and guide:
                                    local_related.append(guideword[1:-1])
                                    global_related.append(guideword[1:-1])
                        local_related = list(set(local_related))
                        global_related = list(set(global_related))
                        
                        temp = []
                        for item in queue:
                            brt_word = item[1]
                            definitions = cam_map[brt_word]
                            sorted_sim_sense = cal_sim_scores(pos, category, local_related,
                                                              global_related, definitions, 
                                                              way, guide)
                            score = float(sorted_sim_sense[0][1])
                            temp.append([score, brt_word])
                        queue = temp
    with open(filename, 'w') as f:
        for data in map_data:
            f.write(json.dumps(data))
            f.write('\n')

#     print('comparison: ', write_file)

#     with open(filename) as f:
#         taste_top = [json.loads(line) for line in f.readlines()]

#     with open('../data/BRT.super.disamb.tsv') as f:
#         rows = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

#         for row in rows:
#              for data in taste_top:
#                 if row[0] == data['super_group'] and \
#                 row[2] == data['category'] and \
#                 row[3] == 'noun' and row[3] == data['pos'] and \
#                 row[4] == data['group'] and \
#                 row[5] == data['brt_word']:
#                     ans = ''
#                     for i in row[6]:
#                         if i == '(':
#                             break
#                         ans += i
#                     word_id = ans.replace('-', '.').split('.')
#                     word_id[-1] = word_id[-1].zfill(2)
#                     word_id = '.'.join(word_id)
#                     if word_id != data['word_id']:
#                         with open(write_file, 'a') as f:
#                             f.write(data['category'] + ' ' + row[4] +'\n')
#                             f.write(word_id + ' ' + row[-2]+'\n')
#                             f.write(data['word_id'] + ' ' + data['en_def'] + '\n')
                            
if __name__ == '__main__':
    main()