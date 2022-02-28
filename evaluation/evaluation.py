import spacy
import argparse
from sentence_transformers import util
from utils import *

def process(sent, SBERT, spacy_model, targetword, token_score, 
            word2defs, emb_map, def2guideword, reserve, 
            sentence_only, topic_only, reweight):
    
    # add guideword in front of sense
    guide_def = gen_guide_def(spacy_model, targetword, word2defs, def2guideword)

    # calculate sentence and definitions similarity 
    def_sent_score = calculate_def_sent_score(sent, guide_def, SBERT)

    if sentence_only:
        sorted_score = sorted(def_sent_score.items(), key=lambda x: x[1], reverse=True)
        return sorted_score
    
    # calculate cosine similarity score between topics and definitions
    weight_score = {}
    for sense in guide_def:
        embs = SBERT.encode(sense, convert_to_tensor=True)
        max_score = float('-inf')
        max_confidence = float('-inf')
        max_topic = ''
        for topic in token_score:
            if topic in emb_map:
                cosine_scores = util.pytorch_cos_sim(embs, emb_map[topic])
                for j in range(len(emb_map[topic])):
                    score = rescal_cos_score(cosine_scores[0][j].cpu())
                    if score > max_score:
                        max_score = score
                        max_topic = topic
                        max_confidence = token_score[topic]
                    elif score == max_score and token_score[topic] > max_confidence:
                        max_topic = topic
                        max_confidence = token_score[topic]
        
        if reweight:
            confidence =  reweight_prob(token_score[max_topic])
        else:
            confidence =  token_score[max_topic]

        if sentence_only:
            weight_score[sense] = def_sent_score[sense]
        elif topic_only:
            weight_score[sense] = confidence * max_score
        else:
            weight_score[sense] = confidence * max_score + (1-confidence) * def_sent_score[sense]
    
    result = sorted(weight_score.items(), key=lambda x: x[1], reverse=True)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-m', type=str, default='brt/cambridge_False_10epochs')
    parser.add_argument('-r', type=int, default=0)  # reserve target word
    parser.add_argument('-s', type=int, default=0)  # sentence only, default False
    parser.add_argument('-mfs', type=int, default=0) # most frequent sense
    parser.add_argument('-t', type=int, default=0)  # topic_only
    parser.add_argument('-w', type=int, default=0)  # reweight
    parser.add_argument('-sm', type=str, default='all-roberta-large-v1') # sbert_model
    
    args = parser.parse_args()
    # parse arguments
    filetype, topic_model_name, mfs_bool, topic_only, \
        reweight, reserve, sentence_only, sbert_model = parse_argument(args)
    
    print_info(topic_model_name, mfs_bool, topic_only, reweight,
                reserve, sentence_only, sbert_model)

    # load data, sbert, spacy, emb_map
    evaluation_data, word2defs, def2guideword = load_data(filetype)
    SBERT, spacy_model = load_spacy_sbert(sbert_model)
    emb_map = load_topic_emb(sbert_model)
    
    if filetype in ['100', '200', '300']:
        sent2ans = load_ans(filetype)

    # if mfs then we need to load ans, else we need to load MLM 
    if mfs_bool:
        first_sense = load_mfs_data()
    else:
        first_sense = ''
        MLM = load_model(topic_model_name)

    save_file = gen_save_filename(sentence_only, mfs_bool, topic_only,
                                    topic_model_name, filetype, reweight,
                                        sbert_model, 'origin')
        
    process_evaluation_data(evaluation_data, filetype, sent2ans, first_sense,
                            mfs_bool, topic_only, reweight, reserve, sentence_only, 
                            save_file, 
                            MLM, SBERT, spacy_model, 
                            word2defs, emb_map, def2guideword,
                            process)

if __name__ == '__main__':
    main()