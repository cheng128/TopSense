import pdb
import argparse
from collections import defaultdict
from sentence_transformers import util
try:
    from .utils import *
except:
    from utils import *

def cal_weighted_score(token_score, emb_map, guide_def, SBERT, reweight, topic_only, def_sent_score):
    weight_score = defaultdict(lambda: 0)
    for topic, confidence in token_score.items():
        if topic in emb_map:
            topic_emb = emb_map[topic].cuda()
            for sense in guide_def:
                sense_emb = SBERT.encode(sense, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(sense_emb, topic_emb)
                sorted_scores = sorted(cosine_scores[0].cpu(), reverse=True)
                top3_scores = [rescale_cos_score(score) for score in sorted_scores[:3]]
                if reweight:
                    confidence =  reweight_prob(token_score[topic])
                else:
                    confidence =  token_score[topic]
                if topic_only:
                    sense_score = confidence * sum(top3_scores) / len(top3_scores)
                else:   
                    sense_score = confidence * sum(top3_scores) / len(top3_scores) +\
                                     (1 - confidence) * def_sent_score[sense]
                weight_score[sense] += sense_score 
                
    for key, value in weight_score.items():
        weight_score[key] = value / len(token_score)
        
    result = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    return result


def formula(sent, SBERT, spacy_model, targetword, token_score, 
            word2defs, emb_map, def2guideword, reserve, 
            sentence_only, topic_only, reweight):
    
    # add guideword in front of sense
    guide_def = gen_guide_def(spacy_model, targetword, word2defs, def2guideword)

    # calculate sentence and definitions similarity
    def_sent_score = calculate_def_sent_score(sent, guide_def, SBERT)

    if sentence_only:
        sorted_score = sorted(def_sent_score.items(), key=lambda x: x[1], reverse=True)
        return sorted_score

    result = cal_weighted_score(token_score, emb_map, guide_def, SBERT, reweight, topic_only, def_sent_score)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str) # evaluation data: 100, 200, 300
    parser.add_argument('-m', type=str, default='brt/cambridge_False_10epochs')
    parser.add_argument('-r', type=int, default=0)  # reserve target word
    parser.add_argument('-s', type=int, default=0)  # sentence only, default False
    parser.add_argument('-mfs', type=int, default=0) # most frequent sense
    parser.add_argument('-t', type=int, default=0)  # topic_only
    parser.add_argument('-w', type=int, default=0)  # reweight
    parser.add_argument('-sm', type=str, default='sentence-t5-xl') # sbert_model
    
    args = parser.parse_args()
    filetype, topic_model_name, mfs_bool, topic_only,\
        reweight, reserve, sentence_only, sbert_model = parse_argument(args)
    
    print_info(topic_model_name, mfs_bool, topic_only, reweight, 
                reserve, sentence_only, sbert_model)
    
    # load data, sbert, spacy, emb_map
    evaluation_data, word2defs, def2guideword = load_data(filetype)
    SBERT, spacy_model = load_spacy_sbert(sbert_model)
    emb_map = load_topic_emb(sbert_model)
    
    if filetype in ['100', '200', '300']:
        sent2ans = load_ans(filetype)
    else:
        sent2ans = defaultdict(lambda:' ')
    # if mfs then we need to load ans, else we need to load MLM 
    if mfs_bool:
        first_sense = load_mfs_data()
    else:
        first_sense = ''
        MLM = load_model(topic_model_name)
    
    save_file = gen_save_filename(sentence_only, mfs_bool, topic_only, reserve,
                                    topic_model_name, filetype, reweight,
                                        sbert_model, 'formula')

    process_evaluation_data(evaluation_data, filetype, sent2ans, first_sense,
                            mfs_bool, topic_only, reweight, reserve, sentence_only, 
                            save_file, 
                            MLM, SBERT, spacy_model, 
                            word2defs, emb_map, def2guideword,
                            formula) # calculate function

if __name__ == '__main__':
    main()