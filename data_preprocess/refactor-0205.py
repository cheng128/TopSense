# TODO:
# 1. add words that not in cambridge and word we handled into local related list
# 2. seperate calculate word and word similarity and sentence to sentence similarity
# 3. global_related -> global one sense

"""
do not calculate similarity between guide word and category then / 2
"""

import csv
import json
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
    
    def init_args(self, args):
        self.args_way = args.c
        self.args_threshold = args.r
        self.args_type = args.f
        self.args_word = args.w
        self.args_guideword = args.g
    
    def init_readfile(self):
        if self.args_type == 'test':
            word = self.args_word
            self.brt_file = f'../data/remap_test_data/BRT_data_{word}_test.json'
        elif self.args_type == 'normal':
            self.brt_file = f'../data/BRT_data.json'
    
    def init_savefile(self):
        jsonl_base = '../data/jsonl_file/'
        compare_base = '../data/comparison/'
        filename = f'{self.args_guideword}_{self.args_threshold}_{self.args_way}'
        if self.args_type == 'test':
            self.jsonl_file = f'{jsonl_base}refactor0205_{filename}_{self.args_word}_test.jsonl'
            self.write_name = f'{compare_base}refactor0205_{filename}_{self.args_word}_test.txt'
        elif self.args_type == 'normal':
            self.jsonl_file = f'{jsonl_base}refactor_{filename}_brt_map_cam.jsonl'
            self.write_name = f'{compare_base}refactor_{filename}_{self.args_word}.txt'
        
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
        print('write file:', self.write_name)
        
    def init_word_related(self):
        related = []
        if self.args_guideword:
            # TODO: add super group?
            supergroup = self.supergroup.lower()
            category = self.category.lower()
            related = [supergroup, category]
        return related
    
    def init_single_sense(self):
        print('init_single_sense')
        single_sense = []
        single_sense_word = []
        for group, words in self.groups.items():
            for word in words:
                definitions = self.cam_map.get(word, {})
                if len(definitions) == 1 and definitions[0]['pos'] == self.pos:
                    single_sense.append(definitions[0]['en_def'])
#                 add words that are not in cambridge dictionary
                elif not definitions:
                    single_sense_word.append(word)
                
        return single_sense, single_sense_word
        
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
            self.local_sense_related.append(sense['en_def'])
            guideword = sense['guideword'].lower()
            if guideword and self.args_guideword:
                self.local_word_related.append(guideword[1:-1])
            # TODO: dedup or not?
            self.local_sense_related = list(set(self.local_sense_related))
            self.local_word_related = list(set(self.local_word_related))
    
#     def cosine_similarity(self, a, b):
#         vec = self.model.encode(a + b, convert_to_tensor=True)
#         cosine_scores = util.cos_sim(vec, vec)
#         edge = len(a)
#         pairs = []
#         for i in range(edge):
#             for j in range(edge, len(cosine_scores)):
#                 pairs.append(cosine_scores[i][j])
#         return pairs

    def cosine_similarity(self, vec, edge):
        cosine_scores = util.cos_sim(vec, vec)
        pairs = []
        for i in range(edge):
            for j in range(edge, len(cosine_scores)):
                pairs.append(cosine_scores[i][j])
        return pairs
    
    def encode(self, sentences, words):
        vec = self.model.encode(sentences + words, convert_to_tensor=True)
        sent_embeds, word_embeds = vec[:len(sentences)], vec[len(sentences):]
        return sent_embeds, word_embeds

    def cal_similarity(self, definitions, sense_related, word_related):
        print('cal_similarity')
        way = self.args_way
        sense_score = []
        for sense in definitions:
            pairs = []
            if sense['pos'] == self.pos:
                sentences = [sense['en_def']]
                sense_words = []

                if sense['examples']:
                    examples = [i['en'] for i in sense['examples']]
                    sentences.extend(examples)

                if sense['guideword'] and self.args_guideword:
                    sense_words.append(sense['guideword'][1:-1].lower())
                
                
                if sense_words:
                    words = sense_words + word_related
                else:
                    words = []
                sent_embeds, word_embeds = self.encode(sentences + sense_related, words)
                sent_scores = self.cosine_similarity(sent_embeds, len(sentences))
                word_scores = self.cosine_similarity(word_embeds, len(sense_words))
                pairs += sent_scores
                pairs += word_scores

#                 pairs.extend(self.cosine_similarity(sentences, sense_related))
#                 if sense_words:
#                     pairs.extend(self.cosine_similarity(sense_words, word_related))
                
                pairs = sorted(pairs, reverse=True)
                
                if way == 'avg':
                    similarity = sum([p for p in pairs]) / len(pairs)
                elif way == 'top3':
                    scores = [p for p in pairs][:3]
                    similarity = sum(scores) / len(scores)
                sense_score.append([sense, similarity])
                
        return sense_score

    def cal_sense_score(self, definitions):
        sense_related = self.local_sense_related + self.global_single_sense
        word_related = self.local_word_related + self.global_single_sense_word
        
#         if not local_word_related and not global_word_sense:
#             # TODO: add super group?
#             local_word_related = [self.category]
        
        # first considerate words that is not ambiguate or in the same group
        sim_scores = self.cal_similarity(definitions, 
                                         sense_related, 
                                         word_related)
        
        sorted_sense_score = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_sense_score
    
    # TODO: remove 
    def monosemous_word(self):
        print('monosemous_word')
        for word in self.words:
            self.word = word
            definitions = self.cam_map.get(word, {})
            if len(definitions) == 1 and definitions[0]['pos'] == self.pos:
                sense = definitions[0]
                score = 1
                store_data = self.gen_store_data(word, sense, score)
                self.map_data.append(store_data)
                # update related list
                self.seen.append(word)
    
    def polysemous_word(self):
        print('polysemous_word')
        for word in self.words:
            if word not in self.seen:
                definitions = self.cam_map.get(word, {})
                if definitions:
                    sorted_sense_score = self.cal_sense_score(definitions)
                    if sorted_sense_score:
                        sense = sorted_sense_score[0][0]
                        score = float(sorted_sense_score[0][1])
                        self.queue.append([score, word, sense])
    
    def clear_queue(self):
        while self.queue:
            print(len(self.queue))
            first_item, self.queue = self.find_max(self.queue)
            score, word, sense = first_item
            store_data = self.gen_store_data(word, sense, score)
            self.map_data.append(store_data)
            self.update_related(sense, score)

            # update similarity score of all definitions in queue to rerank the queue
            temp = []
            for item in self.queue:
                word = item[1]
                definitions = self.cam_map[word]
                sorted_sense_score = self.cal_sense_score(definitions)
                sense = sorted_sense_score[0][0]
                score = float(sorted_sense_score[0][1])
                temp.append([score, word, sense])
            self.queue = temp
                
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
                print(category)
                self.category = category
                for pos, groups in pos_data.items():
                    print(pos)
                    if pos in ['cat_num', 'interjections']:
                        continue
                    self.pos = self.pos_map[pos]
                    self.groups = groups
                    # extract all monosemous word sense
                    self.global_single_sense, self.global_single_sense_word =\
                                    self.init_single_sense()
                    for group, words in groups.items():
                        print('group')
                        self.queue = []
                        self.group = group
                        self.words = words
                        self.local_word_related = self.init_word_related()
                        self.local_sense_related = []
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
    
    REMAP = Remap(args)
    REMAP.run()
    
    
if __name__ == '__main__':
    main()