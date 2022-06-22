import json
from collections import defaultdict

class Evaluator:
    def __init__(self, args):
        self.trained_model = args.m
        self.mfs_bool = bool(args.mfs)
        self.topic_only = bool(args.t)
        self.reweight = bool(args.w)
        self.reserve = bool(args.r)
        self.sentence_only = bool(args.s)
        self.sbert_name = args.sm
        self.pos_tag = args.pos
        self.mark = args.a

    def print_info(self):
        print('predict topic model:', self.trained_model)
        print('Is most frequent sense: ', self.mfs_bool)
        print('Is topic only: ', self.topic_only)
        print('Is reweight: ', self.reweight)
        print('Is reserve: ', self.reserve)
        print('Is sentence_only: ', self.sentence_only)
        print('Sbert model:', self.sbert_name)
        print('POS tag:', self.pos_tag)

    def json_load(self, filename):
        with open(filename) as f:
            data = json.loads(f.read())
        return data

    def load_mfs_data(self):
        filename = f'./TopSense/evaluation/data/{self.pos_tag}_first_sense.json'
        mfs_data = self.json_load(filename)
        return mfs_data 
        
    def load_data(self):
        base_dir = './TopSense/evaluation/data'
        filename = f'{base_dir}/{self.pos_tag}_sentences.json'    
        print('load data: ', filename)
        evaluation_data = self.json_load(filename)
            
        filename = f'{base_dir}/{self.pos_tag}_sentences_ans.json'
        sent2ans = self.json_load(filename)

        first_sense = self.load_mfs_data() if self.mfs_bool else ''
        return evaluation_data, sent2ans, first_sense 

    def gen_save_filename(self):
        trained_model = self.trained_model
        pos = self.pos_tag
        reserve = self.reserve
        sbert = self.sbert_name
        reweight = self.reweight
        mark = self.mark

        directory = trained_model.split('/')[-2]
        model_name = trained_model.split('/')[-1]
        
        prefix = f'./TopSense/evaluation/results/{directory}'
        if self.sentence_only:
            save_file = f'{prefix}/{pos}_sentence_only_{sbert}_{mark}.tsv'
        elif self.mfs_bool:
            save_file = f'{prefix}/{pos}_most_frequent.tsv'
        elif self.topic_only:
            save_file = f'{prefix}/{model_name}_{pos}_reweight{reweight}_reserve{reserve}_topicOnly_{sbert}_{mark}.tsv'
        else:
            save_file = f'{prefix}/{model_name}_{pos}_reweight{reweight}_reserve{reserve}_{sbert}_{mark}.tsv'
        print('save file: ', save_file)

        with open(save_file, 'w') as f:
            pass
        return save_file

    def process_mfs_sense(self, first_sense, targetword, ans, top_one, input_sent, save_filename):
        mfs_sense = first_sense[targetword]
        if ans.endswith(mfs_sense):
            top_one += 1
        senses = [mfs_sense]
        self.write_data(input_sent, targetword, ans, senses, [], save_filename)
        return top_one

    def write_data(self, input_sent, targetword, ans, senses, topics, save_file):
        if self.sentence_only or self.mfs_bool:
            write_data = input_sent + '\t' + targetword + '\t' +\
            ans + '\t' + '\t'.join(senses) + '\n'
        else:
            if len(topics) != 5:
                topics += [''] * (5 - len(topics))
        
            write_data = input_sent + '\t' + targetword + '\t' +\
            '\t'.join(topics) + '\t' + ans + '\t' + '\t'.join(senses) + '\n'

        with open(save_file, 'a') as f:
            f.write(write_data)

    def write_score(self, save_file, rank_score, top_one, total_count):
        with open(save_file, 'a') as f:
            if rank_score:
                f.write('MRR: ' + str(rank_score / total_count) + '\n')
            f.write('Top 1 accuracy: ' + str(top_one / total_count)) 