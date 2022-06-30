"""
Disambiguation Steps:
1. Use model to predict topics
2. Generate topic token score -> dict
3. Calculate cosine similarity between input sentence and candidate senses
4. Calculate cosine similarity between topic and candidate senses
5. return sorted senses

Input and Output for the Main Disambiguation Function:
Please see data_description.txt for more details
"""
import torch
import numpy as np
import torch.nn as nn
from transformers import pipeline
from collections import defaultdict
from .util import gen_masked_sent
from sentence_transformers import util

softmax = nn.Softmax(dim=0)

class Disambiguator:
    
    def __init__(self, word2pos_defs, topic_embs, sense_examples_embs,
                 sbert_model, trained_model_name, tokenizer_name,
                 reserve, sentence_only, reweight, topic_only):
        self.trained_model_name = trained_model_name
        self.tokenizer_name = tokenizer_name
        self.reserve = reserve
        self.sentence_only = sentence_only
        self.reweight = reweight
        self.topic_only = topic_only

        self.word2pos_defs = word2pos_defs
        self.topic_embs_map = topic_embs
        self.sense_examples_embs = sense_examples_embs
        self.SBERT = sbert_model

        self.MLM = self.load_trained_model()

    def load_trained_model(self):
        MLM = pipeline('fill-mask',
                        model=self.trained_model_name, 
                        tokenizer=self.tokenizer_name)
        return MLM

    def reweight_prob(self, prob):
        complement_prob = 1 - prob
        weight = torch.sigmoid(torch.tensor(prob))
        reweighted_probs = softmax(torch.tensor([(prob + weight), complement_prob]))
        return float(reweighted_probs[0])

    def rescal_cos_score(self, score):
        rescale_score = 1 - np.arccos(min(float(score), 1)) / np.pi
        return rescale_score

    def calculate_def_sent_score(self, targetword, pos_tag, input_sentence):
        sent_emb = self.SBERT.encode(input_sentence, convert_to_tensor=True)
        definition_embs = self.sense_examples_embs[(targetword, pos_tag)]['embs']
        definitions = self.sense_examples_embs[(targetword, pos_tag)]['senses']

        # cos_scores = util.pytorch_cos_sim(embs, embs)
        cos_scores = util.pytorch_cos_sim(sent_emb, definition_embs)
        assert(len(cos_scores[0]) == len(definitions))
        def_sent_score = {}
        for j in range(0, len(cos_scores[0])):
            def_sent_score[definitions[j]] = self.rescal_cos_score(cos_scores[0][j])
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
        pos_tag = 'noun' if pos_tag.lower() in ['propn', 'pron'] else pos_tag.lower()
        definitions = self.word2pos_defs[targetword].get(pos_tag, '')
        
        # only one sense, we don't need to disambiguate
        if len(definitions) == 1:
            return [[definitions[0], 1]]
        elif not definitions: 
            return []
        def_sent_score = self.calculate_def_sent_score(targetword, pos_tag, input_sentence)
        if self.sentence_only:
            sorted_score = sorted(def_sent_score.items(), key=lambda x: x[1], reverse=True)
            return sorted_score
        weight_scores = defaultdict(lambda: 0)
        for topic, confidence in token_scores.items():
            topic_emb = self.topic_embs_map.get((topic, pos_tag), None)
            if topic_emb != None:
                for sense in definitions:
                    index = self.sense_examples_embs[(targetword, pos_tag)]['senses'].index(sense)
                    sense_emb = self.sense_examples_embs[(targetword, pos_tag)]['embs'][index]
                    sense_score = self.cal_weighted_score(confidence, topic_emb, sense_emb, 
                                                            def_sent_score, sense)
                    weight_scores[sense] += sense_score 
        for key, value in weight_scores.items():
            weight_scores[key] = value / len(token_scores)
            
        result = sorted(weight_scores.items(), key=lambda x: x[1], reverse=True)
        return result  

    def gen_token_scores(self, mlm_results):
        token_score = {}
        for idx, r in enumerate(mlm_results):
            if r['token_str'].startswith('['):
                topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
                token_score[topic.lower()] = r['score']
        return token_score

    def predict_and_disambiguate(self, sent_tokens, input_sentence, pos_tag, targetword):
        if targetword in self.word2pos_defs:
            masked_sent, _ = gen_masked_sent(sent_tokens, pos_tag, targetword, self.reserve)
            if not masked_sent:
                return '', '', ''
            predicted_results = self.MLM(masked_sent)
            token_scores = self.gen_token_scores(predicted_results)
            ranked_senses = self.disambiguate(targetword, pos_tag, input_sentence, token_scores)
            if ranked_senses:
                return ranked_senses, masked_sent, token_scores
        return '', '', ''