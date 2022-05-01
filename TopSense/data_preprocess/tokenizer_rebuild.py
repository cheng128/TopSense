import json
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict, Counter

tokenizer_name = "../tokenizer_casedFalse"
nlp = pipeline('fill-mask',
       model='../model/hybrid/wiki_reserve_20_True_4epochs_1e-05',
       tokenizer=tokenizer_name)

token_sents = defaultdict(list)

def find_token(filename):
    with open(filename) as f:
        for line in tqdm(f.readlines()):
            split_data = line.split('\t')
            masked_sent = split_data[-1]
            topic_sent = split_data[1]
            try:
                if nlp(masked_sent)[0]['score']>0.5:
                    token_sents[nlp(masked_sent)[0]['token']].append(topic_sent)
            except:
                continue

def find_topic(sent):
    start, end = -1, -1
    for idx, s in enumerate(sent):
        if s == '[':
            start = idx
        if s == ']':
            end = idx
    if start != -1 and end != -1:
        return sent[start:end+1]
    else:
        return ''
                
def main():
    find_token('../data/training_data/0.0_True_noun_cambridge.tsv')
    find_token('../data/training_data/reserve/reremap_20_simple_True.tsv')
    
    num_topic = defaultdict(list)

    for num, sents in token_sents.items():
        for sent in sents:
            topic = find_topic(sent)
            if topic:
                num_topic[num].append(topic)

    for num, topics in num_topic.items():
        num_topic[num] = Counter(topics)
        
    with open('all_tokenizer_rebuild.json', 'w') as f:
        f.write(json.dumps(num_topic))
    
if __name__ == '__main__':
    main()