import json
from tqdm import tqdm
from utils import handle_examples

def load_data():
    with open('../data/wiki/wiki_word_id2sents.json') as f:
        word_id2sents = json.loads(f.read())
    
    with open('../data/remap.word_id.topics.examples.json') as f:
        word_id2topics = json.loads(f.read())
    
    return word_id2sents, word_id2topics

def main():
    word_id2sents, word_id2topics = load_data()
    
    with open('../data/training_data/wiki_link_sent_masked.tsv', 'a') as f:
        for key, value in tqdm(word_id2sents.items()):
            if key in word_id2topics:
                topics = word_id2topics[key]['topics']
                headword = value[0]
                sentences = value[-1]
                for topic in list(set(topics)):
                    topic = '['+ topic +']'
                    for sent_en in sentences:
                        sent_en = sent_en.replace('\n', '')
                        topic_sent, masked_sent = handle_examples(headword, 
                                                                  sent_en, 
                                                                  topic)
                        if topic_sent and masked_sent:
                            f.write('\t'.join([sent_en, topic_sent, 
                                               masked_sent]) + '\n')

if __name__ == '__main__':
    main()