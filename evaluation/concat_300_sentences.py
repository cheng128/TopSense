import json

def load_data():
    with open('./data/100_sentences.json') as f:
        hundred_sentences = json.load(f)
        
    with open('./data/200_sentences.json') as f:
        two_hundred_sentences = json.load(f)
        
    with open(f'./data/100_sentences_ans.json') as f:
        hundred_ans_dict = json.load(f)
        
    with open(f'./data/200_sentences_ans.json') as f:
        two_hundred_ans_dict = json.load(f)
        
    return hundred_sentences, two_hundred_sentences, hundred_ans_dict, two_hundred_ans_dict

def main():
    
    hundred_sentences, two_hundred_sentences, hundred_ans_dict, two_hundred_ans_dict = load_data()
    
    hundred_ans_dict.update(two_hundred_ans_dict)
    
    for word, sents in two_hundred_sentences.items():
        hundred_sentences[word].extend(sents)
        
    with open('./data/300_sentences.json', 'w') as f:
        json.dump(hundred_sentences, f)
        
    with open('./data/300_sentences_ans.json', 'w') as f:
        json.dump(hundred_ans_dict, f)
    
    return

if __name__ == '__main__':
    main()