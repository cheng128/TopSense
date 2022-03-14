# refactpr by Diane

"""
do not calculate similarity between guide word and category then / 2
"""

# TODO:
# 1. add words that not in cambridge and word we handled into local related
# 2. seperate calculate word and word similarity and sentence to sentence similarity
# 3. global_related -> global one sense

import csv
import json
import time
import argparse
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-t5-xl')

class Remap():
    
    def __init__(self, args):
        self.init_args(args)
        self.init_readfile()
        self.init_savefile()
        self.load_data()
        self.print_info()
        
        self.pos_map = {'adj': 'adjective', 'adv':'adverb',
                        'noun': 'noun', 'verb': 'verb'}
#         self.model = SentenceTransformer('all-roberta-large-v1')
        self.model = SentenceTransformer('sentence-t5-xl')
    
    def init_args(self, args):
        self.args_way = args.c
        self.args_threshold = args.r
        self.args_type = args.f
        self.args_word = args.w
        self.args_guideword = True if args.g else False
    
    def init_readfile(self):
        if self.args_type == 'test':
            word = self.args_word
            self.brt_file = f'../data/remap_test_data/BRT_data_{word}_test.json'
        elif self.args_type == 'normal':
            self.brt_file = f'../data/BRT_data.json'
    
    def init_savefile(self):
        jsonl_base = '../data/jsonl_file/'
        filename = f'{self.args_guideword}_{self.args_threshold}_{self.args_way}'
        if self.args_type == 'test':
            self.jsonl_file = f'{jsonl_base}test_jsonl/refactor_{filename}_{self.args_word}_test_cross_reference.jsonl'
        elif self.args_type == 'normal':
            self.jsonl_file = f'{jsonl_base}refactor_{filename}_map_cross_reference.jsonl'
        
    def load_data(self):
        with open(self.brt_file) as f:
            self.brt_data = json.loads(f.read())

        with open('../data/cambridge.sense.000.jsonl') as f:
            cambridge = [json.loads(line) for line in f.readlines()]

        self.cam_map = defaultdict(list)
        for line in cambridge:
            self.cam_map[line['headword']].append(line)
    
    def print_info(self):
        print('file type:', self.args_type)
        print('calculate way:', self.args_way)
        print('threshold:', self.args_threshold)
        print('----------')
        print('read brt file:', self.brt_file)
    
    def gen_store_data(self, word, sense, score):
        data = {'super_group': self.supergroup,
                'category': self.category,
                'pos': sense['pos'],
                'brt_word': word,
                'group': self.group,
                'word_id': sense['id'],
                'en_def': sense['en_def'],
                'ch_def': sense['ch_def'],
                'score': float(score)}
        return data
    
    def write_file(self):
        self.map_data = sorted(self.map_data, key=lambda x: x['group'])
        with open(self.jsonl_file, 'w') as f:
            for data in self.map_data:
                f.write(json.dumps(data))
                f.write('\n')

                            
    def update_related(self, sense, score):
        
        if score >= self.args_threshold:
            self.group_related.append(sense['en_def'])
            guideword = sense['guideword'].lower()
            if guideword and self.args_guideword:
                self.group_related.append(guideword[1:-1])
            self.group_related = list(set(self.group_related))
    
    def cal_similarity(self, sense):
        
        way = self.args_way
        
        sentences = [sense['en_def']]
        sentences.extend(self.related_list)
        sentences.extend(self.group_related)
        
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)

        pairs = [cosine_scores[0][j] for j in range(1, len(cosine_scores))]
        pairs = sorted(pairs, reverse=True)
        top3_scores = pairs[:3]
        similarity = sum(top3_scores) / len(top3_scores)
                       
        self.sense2top3scores[(sense['headword'], sense['en_def'])] = top3_scores 

        return similarity
        
    def cal_related(self, sim_scores, definitions):
        for sense in definitions:
            if sense['pos'] == self.pos:
                self.sense = sense
                similarity = self.cal_similarity(sense)
                sim_scores.append([sense, similarity])
        return sim_scores
    
    def cal_sim_scores(self, definitions):
        if not self.related_list:
            self.related_list = [self.category]
        sim_scores = []
        
        if self.related_list:
            sim_scores = self.cal_related(sim_scores, definitions)
            
        sorted_sim_sense = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return sorted_sim_sense
    
    def init_related_list(self, pos_data):
        self.related_list = []
        self.seen = []
        for pos, groups in pos_data.items():
            if pos in ['cat_num', 'interjections']:
                continue
            self.pos = self.pos_map[pos]
            for group, words in groups.items():
                self.group = group
                for word in words:
                    definitions = self.cam_map.get(word, {})
                    if len(definitions) == 1:
                        sense = definitions[0]
                        self.related_list.append(sense['en_def'])
                        if sense['pos'] == self.pos:
                            score = 1
                            store_data = self.gen_store_data(word, sense, score)
                            self.map_data.append(store_data)
                            # update related list
                            self.seen.append([word, self.pos, self.group])
             
        print('after:', len(self.related_list))

    def polysemous_word(self):
        for word in self.words:
            if [word, self.pos, self.group] not in self.seen:
                definitions = self.cam_map.get(word, {})
                if definitions:
                    sorted_sim_sense = self.cal_sim_scores(definitions)
                    if sorted_sim_sense:
                        score = float(sorted_sim_sense[0][1])
                        sense = sorted_sim_sense[0][0]
                        self.queue.append([score, word, sense])
    
    def find_idx(self, last_idx, value, top3_scores):
        while last_idx > -1:
            if value > top3_scores[last_idx]:
                last_idx -= 1
            else:
                break
        return last_idx + 1

    def update_similarity_score(self, definitions, first_sense):

        sentences = [first_sense]
        sense2data = {}
        for sense in definitions:
            if self.pos == sense['pos']:  
                sentences.append(sense['en_def'])
                sense2data[sense['en_def']] = sense 
        try:
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
        except:
            print(sentences)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        
        pairs = {sentences[j]: cosine_scores[0][j] 
                 for j in range(1, len(cosine_scores))}

        headword = definitions[0]['headword']
        max_similarity = float('-inf')
        max_sense = ''

        for sense, value in pairs.items(): 
            top3_scores = self.sense2top3scores[(headword, sense)]
            insert_idx = self.find_idx(len(top3_scores) - 1, value, top3_scores)
            top3_scores.insert(insert_idx, value)
            top3_scores = top3_scores[:3]
            similarity = sum(top3_scores) / len(top3_scores)
            if similarity > max_similarity:
                max_similarity = similarity
                max_sense = sense
        max_sense = sense2data[max_sense]
        return max_similarity, max_sense        

    def clear_queue(self):
        start = time.time()
        
        print('queue length:', len(self.queue))
        while self.queue:
            first_item, self.queue = self.find_max(self.queue)
            score, word, first_sense = first_item
            store_data = self.gen_store_data(word, first_sense, score)
            self.map_data.append(store_data)
            self.update_related(first_sense, score)

            # update similarity score of all definitions in queue to rerank the queue
            temp = []
            for item in self.queue:
                word = item[1]
                definitions = self.cam_map[word]
                score, sense = self.update_similarity_score(definitions, first_sense['en_def'])
                temp.append([score, word, sense])
            self.queue = temp
        end = time.time()
        print(f'{self.pos} {self.group} clear queue spend: {end-start}')
                
    def find_max(self, queue):
        max_item = queue[0]
        temp = []
        for item in queue[1:]:
            if item[0] > max_item[0]:
                temp.append(max_item)
                max_item = item
            else:
                temp.append(item)
        return max_item, temp

    def run(self):
        
        self.map_data = []
        
        start = time.time()
        for supergroup, categories in self.brt_data.items():
            self.supergroup = supergroup
            print(supergroup)
            self.write_file()
            for category, pos_data in tqdm(categories.items()):
                self.category = category
                self.init_related_list(pos_data)
                for pos, groups in pos_data.items():
                    if pos in ['cat_num', 'interjections']:
                        continue
                    self.pos = self.pos_map[pos]
                    for group, words in groups.items():
                        print(f'pos: {pos} group: {group}')
                        self.group_related = []
                        self.queue = []
                        self.sense2top3scores = {}
                        self.group = group
                        self.words = words
                        # put other ponosemous word into queue 
                        self.polysemous_word()
                        self.clear_queue()
        end = time.time()
        print('end of the process:' , end-start)
        self.write_file()
        
def main():
    parser = argparse.ArgumentParser()
    # the way to calculate similarity, top3, top5 or avg
    parser.add_argument('-c', type=str, default='top3')
    # the threshold of add a sentence into local, global related
    parser.add_argument('-r', type=float, default=0)
    # file type = 'test' or 'normal'
    parser.add_argument('-f', type=str, default='test')
    # if it is a test, then we can use a word to test the result
    parser.add_argument('-w', type=str, default='all')
    # add guideword into local, global related
    parser.add_argument('-g', type=int, default=1)
    args = parser.parse_args()
    
    REMAP = Remap(args)
    REMAP.run()
    
    
if __name__ == '__main__':
    main()