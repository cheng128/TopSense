import csv
import json
import spacy
from random import sample
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
model = pipeline('fill-mask',
                 model=f"../model/wiki_similarity/10sents_10epochs",
                 tokenizer="../topic_tokenizer")

def fetch_data():
    words = {}
    with open('./data/Yarowsky1992_test.csv') as f:
        csv_reader = csv.DictReader(f)
        for data in csv_reader:
            if data['pos'] == 'noun':
                words[data['word']] = data['ans'].lower()
    return words

def fetch_voa():
    with open('./data/voa_sentences.json') as f:
        voa_sentences = json.loads(f.read())
    return voa_sentences

def calculate_separate(results, answers):
    right_count = 1
    total_count = 1
    numbers = []
    for result in results:
        if result['token_str'][1:-1] in answers:
            numbers.append(right_count/total_count)
            right_count += 1
            total_count += 1
        else:
            total_count += 1
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)

def handle_mask(word, sent):
    doc = nlp(sent)
    sent_tokens = []
    count = 1
    for token in doc:
        if word in [token.text, token.lemma_] and count == 1:
            s = token.text if word == token.text else token.lemma_
            count += 1
            sent_tokens.append(f'[MASK]')
        else:
            sent_tokens.append(token.text)
    sent = ' '.join(sent_tokens)
    return sent

def main():
    key_answers = fetch_data()
    voa_sentences = fetch_voa()
    avg_num = []
    count = len(key_answers)
    for word, answer in key_answers.items():
        count -= 1
        separate_word = []
        print(count)
        if word != 'galley':
            sentences = voa_sentences[word]
#             if len(sentences) > 20:
#                 sentences = sample(sentences, 20)
            for sent in sentences:
                masked_sent = handle_mask(word, sent)
                results = model(masked_sent)
                precision = calculate_separate(results, answer)
                separate_word.append(precision)
            avg_num.append(sum(separate_word)/len(separate_word))

    with open('./results/wiki_similarity_10sent_map_yarowsky_11.txt', 'a') as f:
        f.write(f'map: {sum(avg_num) / len(avg_num)} \n')
        
if __name__ == '__main__':
    main()