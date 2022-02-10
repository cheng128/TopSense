""" This file process BRT.super.disamb.txt data in to following format:
    'word_id' = {"headword":headword, "topics":[list of topics]}
"""

import json
from collections import defaultdict

class Data():
    def __init__(self):
        with open('../data/BRT.super.disamb.tsv') as f:
            self.BRT = [line.split('\t') for line in f.readlines()]

        with open('../data/cambridge.sense.000.jsonl') as f:
            self.cambridge = [json.loads(line) for line in f.readlines()]
            
        self.id_dict = {line['id']: [line['pos'], [p['en'] for p in line['examples']]]
                        for line in self.cambridge}

    def clean_word_id(self, word_id):
        """ we do not want the "(GUIDEWORD)" part in word id
            so we need to cleanup word id.
        """
        pos_dict = {'noun': 'noun', 'verb':'verb', 'adjective':'adj', 'adverb':'adv'}
        no_brackets = ''
        for s in word_id:
            if s == '(':
                break
            else:
                no_brackets += s
        word_id = no_brackets.rsplit('-', 2)
        word_id[-1] = word_id[-1].zfill(2)
        word_id[-2] = pos_dict[word_id[-2]]
        word_id = '.'.join(word_id)
        return word_id
    
    def process(self):
        temp_BRT_dict = defaultdict(list)
        for data in self.BRT:
            word_id = self.clean_word_id(data[6])
            temp_BRT_dict[(data[5], word_id)].append(data[1] + data[2])
        print('temp_dict complete')
        # save all pos of words, topics and their examples
        BRT_dict = {}

        for k, v in temp_BRT_dict.items():
            pos, examples = self.id_dict.get(k[1], [None, None])
            if not pos:
                continue
            if 'Animals' in v:
                print(v)
            BRT_dict[k[1]] = {'headword': k[0], 
                               'topics': list(set(v)),
                               'pos': pos,
                               'examples': examples}
                    
        with open('../data/word_id.topics.examples.json', 'w') as f:
            f.write(json.dumps(BRT_dict))
        
def main():
    MAIN_CLASS = Data()
    MAIN_CLASS.process()
    
if __name__ == '__main__':
    main()