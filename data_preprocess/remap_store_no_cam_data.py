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
        # self.model = SentenceTransformer('all-roberta-large-v1')
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
            self.brt_file = f'../data/remap_test_data/BRT_data_{word}_test_new.json'
        elif self.args_type == 'normal':
            self.brt_file = f'../data/BRT_data.json'
    
    def init_savefile(self):
        jsonl_base = '../data/jsonl_file/'
        filename = f'{self.args_guideword}_{self.args_threshold}_{self.args_way}'
        if self.args_type == 'test':
            self.jsonl_file = f'{jsonl_base}test_jsonl/{filename}_{self.args_word}_not_in_cam.jsonl'
        elif self.args_type == 'normal':
            self.jsonl_file = f'{jsonl_base}{filename}_not_in_cam.jsonl'
        
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
    
    def gen_store_data(self, word, sense, score):
        data = {'super_group': self.supergroup,
                'category': self.category,
                'pos': sense['pos'] if sense else self.pos,
                'brt_word': word,
                'group': self.group,
                'word_id': sense['id'] if sense else '',
                'en_def': sense['en_def'] if sense else '',
                'ch_def': sense['ch_def'] if sense else '',
                'score': float(score)}
        return data
    
    def write_file(self):
        map_data = self.map_data
        # map_data = sorted(map_data, key=lambda x: x['category'])
        # map_data = sorted(map_data, key=lambda x: x['group'])
        with open(self.jsonl_file, 'w') as f:
            for data in map_data:
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
        if self.group_related or self.group_mono_related:
            sentences.extend(self.group_related)
            sentences.extend(self.group_mono_related)
        # elif self.related_list:
            # sentences. extend(self.related_list)
        
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings, embeddings)

        pairs = [cosine_scores[0][j] for j in range(1, len(cosine_scores))]
        scores = sorted(pairs, reverse=True)

        if self.args_way == 'top3':
            scores = scores[:3]            

        similarity = sum(scores) / len(scores)
        
        self.sense2scores[(sense['headword'], sense['en_def'])] = scores

        return similarity
        
    def cal_related(self, sim_scores, definitions):
        for sense in definitions:
            if sense['pos'] == self.pos:
                self.sense = sense
                similarity = self.cal_similarity(sense)
                sim_scores.append([sense, similarity])
        return sim_scores
    
    def cal_sim_scores(self, definitions):
        
        sim_scores = []
        sim_scores = self.cal_related(sim_scores, definitions)
            
        sorted_sim_sense = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return sorted_sim_sense

    def monosemous_word(self):
        for word in self.words:
            definitions = self.cam_map.get(word, {})
            if len(definitions) == 1 and definitions[0]['pos'] == self.pos:
                sense = definitions[0]
                store_data = self.gen_store_data(word, sense, 1)
                self.map_data.append(store_data)
                self.group_mono_related.append(sense['en_def'])
                self.seen.append([word, self.pos, self.group])
            else:
                if len(word.split()) > 1 and word.split()[-1].isdigit():
                    target_word = ' '.join(word.split()[:-1]).lower()
                    self.group_mono_related.append(target_word)
                    self.seen.append([word, self.pos, self.group])
                elif len(definitions) <= 1:
                    if len(word.split()) > 1:
                        store_data = self.gen_store_data(word, {}, 0)
                        self.map_data.append(store_data)
                    self.group_mono_related.append(word)
                    self.seen.append([word, self.pos, self.group])
        self.group_mono_related = list(set(self.group_mono_related))

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
    
    def find_idx(self, value, scores): 
        idx = 0
        while idx < len(scores):
            if value < scores[idx]:
                idx += 1
            else:
                break
        return idx

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
            scores = self.sense2scores[(headword, sense)]
            insert_idx = self.find_idx(value, scores)
            scores.insert(insert_idx, value)
            if self.args_way == 'top3':
                scores = scores[:3]
            similarity = sum(scores) / len(scores)
            if similarity > max_similarity:
                max_similarity = similarity
                max_sense = sense
        max_sense = sense2data[max_sense]
        return max_similarity, max_sense        

    def clear_queue(self):
        start = time.time()
        
        # print('queue length:', len(self.queue))
        while self.queue:
            first_item, self.queue = self.find_max(self.queue)
            score, word, first_sense = first_item
            store_data = self.gen_store_data(word, first_sense, score)
            self.map_data.append(store_data)
            self.update_related(first_sense, score)
            self.group_related = list(set(self.group_related))

            # update similarity score of all definitions in queue to rerank the queue
            temp = []
            for item in self.queue:
                word = item[1]
                definitions = self.cam_map[word]
                score, sense = self.update_similarity_score(definitions, first_sense['en_def'])
                temp.append([score, word, sense])
            self.queue = temp
        end = time.time()
        # print(f'{self.pos} {self.group} clear queue spend: {end-start}')
                
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
                for pos, groups in pos_data.items():
                    if pos in ['cat_num', 'interjections']:
                        continue
                    self.pos = self.pos_map[pos]
                    for group, words in groups.items():
                        self.group_related = [supergroup, category]
                        self.group_mono_related = []
                        self.queue = []
                        self.sense2scores = {}
                        self.group = group
                        self.words = words
                        # put other ponosemous word into queue
                        self.seen = []
                        self.monosemous_word() 
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