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

class Remap():
    
    def __init__(self, args):
        self.init_args(args)
        self.init_readfile()
        self.init_savefile()
        self.load_data()
        self.print_info()
        
        self.pos_map = {'adj': 'adjective', 'adv':'adverb',
                        'noun': 'noun', 'verb': 'verb'}
        self.model = SentenceTransformer('all-roberta-large-v1')
#         self.model = SentenceTransformer('sentence-t5-xl')
    
    def init_args(self, args):
        self.args_way = args.c
        self.args_threshold = args.r
        self.args_type = args.f
        self.args_word = args.w
        self.args_guideword = bool(args.g)
    
    def init_readfile(self):
        if self.args_type == 'test':
            word = self.args_word
            self.brt_file = f'../data/remap_test_data/BRT_data_{word}_test_new.json'
        elif self.args_type == 'normal':
            self.brt_file = f'../data/BRT_data.json'
    
    def init_savefile(self):
        jsonl_base = '../data/jsonl_file/'
        compare_base = '../data/comparison/'
        filename = f'{self.args_guideword}_{self.args_threshold}_{self.args_way}'
        if self.args_type == 'test':
            self.jsonl_file = f'{jsonl_base}/test_jsonl/refactor_{filename}_{self.args_word}_test.jsonl'
        elif self.args_type == 'normal':
            self.jsonl_file = f'{jsonl_base}refactor_{filename}_map.jsonl'
        
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
        print('write jsonl file:', self.jsonl_file)
        
    def init_related(self):
        related = []
        if self.args_guideword:
            supergroup = self.supergroup.lower()
            category = self.category.lower()
            related = [supergroup, category]
        return related
    
    def gen_store_data(self, word, sense, score):
        data = {'super_group': self.supergroup,
                'category': self.category,
                'pos': sense['pos'],
                'brt_word': word,
                'group': self.group,
                'word_id': sense['id'],
                'en_def': sense['en_def'],
                'ch_def': sense['ch_def'],
                'score': score}
        return data
    
    def write_file(self):
        with open(self.jsonl_file, 'w') as f:
            for data in self.map_data:
                f.write(json.dumps(data))
                f.write('\n')
                            
    def update_related(self, sense, score):
        
        if score >= self.args_threshold:
            self.local_related.append(sense['en_def'])
            self.global_related.append(sense['en_def'])
            guideword = sense['guideword'].lower()
            if guideword and self.args_guideword:
                self.local_related.append(guideword[1:-1])
                self.global_related.append(guideword[1:-1])
                self.local_related.append(sense['headword'])
            self.local_related = list(set(self.local_related))
            self.global_related = list(set(self.global_related))
    
    def cal_similarity(self, sense, related):
        way = self.args_way
        
        sentences = [self.sense['en_def']]
        if sense['examples']:
            examples = [i['en'] for i in sense['examples']]
            sentences.extend(examples)
        if sense['guideword'] and self.args_guideword:
            sentences.append(sense['guideword'][1:-1].lower())
        edge = len(sentences)
        sentences.extend(related)
        
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)
        
        pairs = []
        for i in range(edge):
            for j in range(edge, len(cosine_scores)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
        
        if way == 'avg':
            similarity = sum([p['score'] for p in pairs]) / len(pairs)
        elif way == 'top3':
            scores = [p['score'] for p in pairs][:3]
            similarity = sum(scores) / len(scores)
                       
        return similarity
        
    def cal_related(self, definitions, related):
        for sense in definitions:
            if sense['pos'] == self.pos:
                self.sense = sense
                self.sim_scores.append([sense, self.cal_similarity(sense, related)])
    
    def cal_sim_scores(self, definitions):
        local_related = self.local_related
        global_related = self.global_related
        if not local_related and not global_related:
            local_related = [self.category]
        self.sim_scores = []
        
        # first considerate words that is not ambiguate or in the same group
        if local_related:
            self.cal_related(definitions, local_related)
        elif global_related:
            self.cal_related(definitions, global_related)
        sorted_sim_sense = sorted(self.sim_scores, key=lambda x: x[1], reverse=True)
        return sorted_sim_sense
    
    def monosemous_word(self):
        for word in self.words:
            self.word = word
            definitions = self.cam_map.get(word, {})
            if len(definitions) == 1 and definitions[0]['pos'] == self.pos:
                sense = definitions[0]
                score = 1
                store_data = self.gen_store_data(word, sense, score)
                self.map_data.append(store_data)
                # update related list
                self.update_related(sense, score)
                self.seen.append(word)
    
    def polysemous_word(self):
        for word in self.words:
            if word not in self.seen:
                definitions = self.cam_map.get(word, {})
                if definitions:
                    sorted_sim_sense = self.cal_sim_scores(definitions)
                    if sorted_sim_sense:
                        score = float(sorted_sim_sense[0][1])
                        self.queue.append([score, word])
    
    def clear_queue(self):
        start = time.time()

        print('queue length:', len(self.queue))
        while self.queue:
            first_item, self.queue = self.find_max(self.queue)
            word = first_item[1]
            # TODO: do not need to calculate again
            definitions = self.cam_map[word]
            sorted_sim_sense = self.cal_sim_scores(definitions)
            if sorted_sim_sense:
                sense = sorted_sim_sense[0][0]
                score = float(sorted_sim_sense[0][1])
                store_data = self.gen_store_data(word, sense, score)
                self.map_data.append(store_data)
                self.update_related(sense, score)

            # update similarity score of all definitions in queue to rerank the queue
            temp = []
            for item in self.queue:
                word = item[1]
                definitions = self.cam_map[word]
                sorted_sim_sense = self.cal_sim_scores(definitions)
                score = float(sorted_sim_sense[0][1])
                temp.append([score, word])
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
        
        for supergroup, categories in self.brt_data.items():
            self.supergroup = supergroup
            print(supergroup)
            self.write_file()
            for category, pos_data in tqdm(categories.items()):
                self.category = category
                for pos, groups in pos_data.items():
                    if pos in ['cat_num', 'interjections']:
                        continue
                    self.global_related = self.init_related()
                    self.pos = self.pos_map[pos]
                    for group, words in groups.items():
                        self.queue = []
                        self.group = group
                        self.words = words
                        self.local_related = self.init_related()
                        self.seen = []
                        # first deal with monosemous word
                        self.monosemous_word() 
                        # put other ponosemous word into queue 
                        self.polysemous_word()
                        
                        self.clear_queue()
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