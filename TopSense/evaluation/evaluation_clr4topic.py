# best version so far: v4_2ep
# modified: add filetype (100, 200, 300), topic only, etc.
# use new formula 

import sys
import pdb
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import pipeline

from utils import *
from evaluation_formula import formula

sys.path.append('../')
from CLR4Topic.CLR4Topic.predictors.topic_predictor import TopicPredictor
from CLR4Topic.CLR4Topic.models.topic_classifier import TopicClassifier
from CLR4Topic.CLR4Topic.dataset_readers.example_reader import ExampleReader


def load_model(model_name):
    path = f'../CLR4Topic/trained_models/{model_name}/model.tar.gz'
    predictor = TopicPredictor.from_path(path,'topic_predictor')
    
    return predictor

def handle_examples(nlp, headword, sent_en):
    reconstruct = []
    doc = nlp(sent_en)
    
    word = ''
    count = 0
    for token in doc:
        find = False
        if token.text.lower() == headword:
            word = token.text
            find = True
        elif token.lemma_.lower() == headword:
            word = token.lemma_
            find = True

        if find and count == 0:
            reconstruct.append(f'[{token.text}]')
            count += 1
            continue
        
        reconstruct.append(token.text)
            
    masked_sent = ' '.join(reconstruct)
    return masked_sent 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-m', type=str, default='clr/v4_2ep')
    parser.add_argument('-r', type=int, default=0)  # reserve target word
    parser.add_argument('-s', type=int, default=0)  # sentence only, default False
    parser.add_argument('-mfs', type=int, default=0) # most frequent sense
    parser.add_argument('-t', type=int, default=0)  # topic_only
    parser.add_argument('-w', type=int, default=0)  # reweight
    parser.add_argument('-sm', type=str, default='all-roberta-large-v1') # sbert_model
    
    args = parser.parse_args()
    filetype, topic_model_name, mfs_bool, topic_only, \
        reweight, reserve, sentence_only, sbert_model = parse_argument(args)
    
    print_info(topic_model_name, mfs_bool, topic_only, reweight, 
                reserve, sentence_only, sbert_model)

    # load data, sbert, spacy emb_map
    evaluation_data, word2defs, def2guideword = load_data(filetype)
    SBERT, spacy_model = load_spacy_sbert(sbert_model)
    emb_map = load_topic_emb(sbert_model)
    
    if filetype in ['100', '200', '300']:
        sent2ans = load_ans(filetype)
    
    if mfs_bool:
        first_sense = load_mfs_data()
    else:
        first_sense = ''
        predictor = load_model(topic_model_name.split('/')[-1])

    save_file = gen_save_filename(sentence_only, mfs_bool, topic_only, reserve,
                                    topic_model_name, filetype, reweight,
                                        sbert_model, 'clr4topic')

    total_count, rank_score, top_one = 0, 0, 0
    for targetword, sentences in tqdm(evaluation_data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype in ['100', '200', '300'] and not ans:
                continue
                
            # if it is most frequent sense
            if mfs_bool:
                mfs_sense = first_sense[targetword]
                if mfs_sense == ans:
                    top_one += 1
                senses = [mfs_sense]
                write_data(mfs_bool, sentence_only, sent, targetword, 
                           ans, senses, [], save_file)
            else:
                input_sent = handle_examples(spacy_model, targetword, sent)
                mlm_results = predictor(input_sent)
                token_score = {}
                for idx, r in enumerate(mlm_results):
                    if r['token_str'].startswith('['):
                        topic = ' '.join(r['token_str'].split(' ')[1:])[:-1]
                        token_score[topic.lower()] = r['score']
                    continue
                topics = list(token_score.keys())

                results = formula(sent, SBERT, spacy_model, targetword, 
                                  token_score, word2defs, emb_map, def2guideword, 
                                  reserve, sentence_only, topic_only, reweight)
                senses = [line[0].replace('\n', '').strip() 
                         for line in results]
                # if senses[0] == ans:
                #     top_one += 1
                # rank_score += 1 / (senses.index(ans) + 1)
                if ans.endswith(senses[0]):
                    top_one += 1
                for idx, sense in enumerate(senses):
                    if ans.endswith(sense):
                        rank_score += 1 / (idx + 1)
                    else:
                        rank_score += 0
                write_data(mfs_bool, sentence_only, input_sent, targetword,
                       ans, senses, topics, save_file)
            total_count += 1

    write_score(save_file, rank_score, top_one, total_count)

if __name__ == '__main__':
    main()