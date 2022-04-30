"""
Disambiguation Steps:
1. Use model to predict topics
2. Generate topic token score -> dict
3. Calculate cosine similarity between 
    input sentence and candidate senses
4. Calculate cosine similarity between topic
    and candidate senses
5. return sorted senses

Input and Output for the Main Disambiguation Function:
- Input
1. sbert_data -> load SBERT model
2. trained_model_name -> load fine-tuned model
3. input sentence -> generate masked sentence and calculate
   similarity between input sentence and sense
3. targetword -> fetch candidate senses and generate masked sentence
4. pos_tag -> fetch candidate senses and generated masked sentence
5. reserve -> generated masked sentence
6. sentence_only, reweight, topic_only -> for disambiguation calculation

- Output

Please see data_description.txt for more details

"""
import torch
import numpy as np
import torch.nn as nn
from transformers import pipeline
from collections import defaultdict
from sentence_transformers import util


softmax = nn.Softmax(dim=0)

class Disambiguator:

    def __init__(self, data_class, trained_model_name,
                reserve, sentence_only, reweight, topic_only):
        self.trained_model_name = trained_model_name
        self.reserve = reserve
        self.sentence_only = sentence_only
        self.reweight = reweight
        self.topic_only = topic_only

        self.word2pos_defs, self.topic_embs_map, self.sense_examples_embs = data_class.load_data()
        self.SBERT = data_class.load_sbert_model()
        self.SPACY = data_class.load_spacy_model()

        self.MLM = self.load_trained_model()

    def load_trained_model(self):
        model_path = f"../model/{self.trained_model_name}"

        tokenizer = "../tokenizer_casedFalse"
        MLM = pipeline('fill-mask', model=model_path, tokenizer=tokenizer)
        return MLM

    def reweight_prob(self, prob):
        complement_prob = 1 - prob
        weight = torch.sigmoid(torch.tensor(prob))
        reweighted_probs = softmax(torch.tensor([(prob + weight), complement_prob]))
        return float(reweighted_probs[0])

    def rescal_cos_score(self, score):
        rescale_score = 1 - np.arccos(min(float(score), 1)) / np.pi
        return rescale_score

    def calculate_def_sent_score(self, input_sentence, definitions):
        # TODO: sense examples embs
        sentence_defs = [input_sentence]
        sentence_defs.extend(definitions)
        embs = self.SBERT.encode(sentence_defs, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(embs, embs)

        def_sent_score = {}
        for j in range(1, len(cos_scores)):
            def_sent_score[sentence_defs[j]] = self.rescal_cos_score(cos_scores[0][j])
        return def_sent_score
    
    def cal_weighted_score(self, confidence, topic_emb, sense_emb, def_sent_score, sense):
        cosine_scores = util.pytorch_cos_sim(sense_emb, topic_emb)
        sorted_scores = sorted(cosine_scores[0], reverse=True)
        top3_scores = [self.rescal_cos_score(score) for score in sorted_scores[:3]]
        if self.reweight:
            confidence =  self.reweight_prob(confidence)
        if self.topic_only:
            sense_score = confidence * sum(top3_scores) / len(top3_scores)
        else:   
            sense_score = confidence * sum(top3_scores) / len(top3_scores) +\
                            (1 - confidence) * def_sent_score[sense]
        return sense_score

    def disambiguate(self, targetword, pos_tag, input_sentence, token_scores):
        pos_tag = 'noun' if pos_tag == 'propn' else pos_tag
        definitions = self.word2pos_defs[targetword][pos_tag][:]
        def_sent_score = self.calculate_def_sent_score(input_sentence, definitions)

        if self.sentence_only:
            sorted_score = sorted(def_sent_score.items(), key=lambda x: x[1], reverse=True)
            return sorted_score

        weight_scores = defaultdict(lambda: 0)
        for topic, confidence in token_scores.items():
            topic_emb = self.topic_embs_map.get(topic, '')
            if topic_emb != '':
                for sense in definitions:
                    sense_emb = self.SBERT.encode(sense, convert_to_tensor=True)
                    sense_score = self.cal_weighted_score(confidence, topic_emb, sense_emb, 
                                                            def_sent_score, sense)
                    weight_scores[sense] += sense_score 
                    
        for key, value in weight_scores.items():
            weight_scores[key] = value / len(token_scores)
            
        result = sorted(weight_scores.items(), key=lambda x: x[1], reverse=True)
        return result

    def gen_masked_sent(self, input_sentence, pos_tag, targetword):

        reconstruct = []
        doc = self.SPACY(input_sentence) 

        find = False
        for token in doc:
            token_pos = 'NOUN' if token.pos_ == 'PROPN' else token.pos_
            if pos_tag.upper() == token_pos and find == False:
                if token.text.lower() == targetword:
                    word = token.text
                    find = True
                elif token.lemma_.lower() == targetword:
                    word = token.lemma_
                    find = True

                if find:
                    if self.reserve:
                        reconstruct.append(word + ' [MASK]')
                    else:
                        reconstruct.append('[MASK]')
                    continue

            reconstruct.append(token.text)

        masked_sent = ' '.join(reconstruct)
        masked_sent = '' if '[MASK]' not in masked_sent else masked_sent
        return masked_sent 

    def gen_token_scores(self, mlm_results):
        token_score = {}
        for idx, r in enumerate(mlm_results):
            if r['token_str'].startswith('['):
                topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
                token_score[topic.lower()] = r['score']
        return token_score

    def predict_and_disambiguate(self, input_sentence, pos_tag, targetword):
        masked_sent = self.gen_masked_sent(input_sentence, pos_tag, targetword)
        if not masked_sent:
            print(input_sentence)
            return '', '', ''
        predicted_results = self.MLM(masked_sent)
        token_scores = self.gen_token_scores(predicted_results)
        ranked_senses = self.disambiguate(targetword, pos_tag, input_sentence, token_scores)
        return ranked_senses, masked_sent, token_scores

        

