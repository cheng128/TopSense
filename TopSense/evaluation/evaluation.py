import argparse
from tqdm import tqdm
from .utils import *
import sys
from ..disambiguator_class import Disambiguator 
from ..data_class import Data
from ..util import tokenize_processor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str) # evaluation data
    parser.add_argument('-m', type=str) # fine-tuned model
    parser.add_argument('-r', type=int, default=0)  # reserve target word
    parser.add_argument('-s', type=int, default=0)  # sentence only
    parser.add_argument('-mfs', type=int, default=0) # most frequent sense
    parser.add_argument('-t', type=int, default=0)  # topic_only
    parser.add_argument('-w', type=int, default=0)  # reweight
    parser.add_argument('-sm', type=str, default='sentence-t5-xl') # sbert_name
    parser.add_argument('-pos', type=str, default='noun')
    parser.add_argument('-a', type=str, default='')
    args = parser.parse_args()
    
    # we can only choose one type from below at a time
    assert((args.mfs + args.s + args.t) <= 1)
    assert(args.f in ['100', '200', '300', 'verb', 'adjective', 'adverb'])

    filetype, trained_model_name, mfs_bool, topic_only, reweight, reserve,\
    sentence_only, sbert_name, pos_tag, mark = parse_argument(args)
    tokenizer_name = './TopSense/tokenizer_casedFalse'

    DATA = Data(sbert_name, './TopSense/data', 'cpu')
    word2pos_defs, topic_embs, sense_examples_embs = DATA.load_data()
    sbert_model = DATA.load_sbert_model()
     
    DISAMBIGUATOR = Disambiguator(word2pos_defs, topic_embs, sense_examples_embs,
                                  sbert_model, trained_model_name, tokenizer_name,
                                  reserve, sentence_only, reweight, topic_only)
    
    print_info(trained_model_name, mfs_bool, topic_only, reweight, 
                reserve, sentence_only, sbert_name, pos_tag)
    
    evaluation_data, sent2ans, first_sense = load_data(filetype, mfs_bool)
    save_filename = gen_save_filename(sentence_only, mfs_bool, topic_only, reserve, trained_model_name, 
                                      filetype, reweight, sbert_name, pos_tag, mark)

    with open(save_filename, 'w') as f:
        pass

    if mfs_bool:
        process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, 
                         sentence_only, sent, targetword, save_filename) 

    total_count, rank_score, top_one = 0, 0, 0
    for targetword, sentences in tqdm(evaluation_data.items()):
        for ans_key, sent in sentences:
            ans = sent2ans.get(ans_key, '')
            print(ans)
            if not ans:
                continue
            # if it is most frequent sense
            if mfs_bool:
                top_one = process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, 
                                            sentence_only, sent, targetword, save_filename)
            else:
                tokens = tokenize_processor(sent)
                results, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                            sent, 
                                                                                            pos_tag, 
                                                                                            targetword)
                if not results and pos_tag == 'noun':
                    results, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                                sent,
                                                                                                'propn', 
                                                                                                targetword)
                
                senses = [line[0].replace('\n', '').strip() for line in results]
                if ans.endswith(senses[0]):
                    top_one += 1
                for idx, sense in enumerate(senses):
                    if ans.endswith(sense):
                        rank_score += 1 / (idx + 1)
                topics = list(token_scores.keys())
                write_data(mfs_bool, sentence_only, masked_sent, targetword,
                            ans, senses, topics, save_filename)
            total_count += 1
            
    write_score(save_filename, rank_score, top_one, total_count)

if __name__ == '__main__':
    main()