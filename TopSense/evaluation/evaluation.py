import argparse
from tqdm import tqdm
from utils import *
import sys
sys.path.append("..")
from disambiguator_class import Disambiguator 
from data_class import Data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str) # evaluation data: 100, 200, 300
    parser.add_argument('-m', type=str, default='brt/cambridge_False_10epochs')
    parser.add_argument('-r', type=int, default=0)  # reserve target word
    parser.add_argument('-s', type=int, default=0)  # sentence only
    parser.add_argument('-mfs', type=int, default=0) # most frequent sense
    parser.add_argument('-t', type=int, default=0)  # topic_only
    parser.add_argument('-w', type=int, default=0)  # reweight
    parser.add_argument('-sm', type=str, default='sentence-t5-xl') # sbert_name
    args = parser.parse_args()

    assert((args.mfs + args.s + args.t) <= 1)

    filetype, trained_model_name, mfs_bool, topic_only,\
    reweight, reserve, sentence_only, sbert_name = parse_argument(args)

    DATA = Data(sbert_name, '../data')
    DISAMBIGUATOR = Disambiguator(DATA, trained_model_name,
                                  reserve, sentence_only, reweight, topic_only)
    
    print_info(trained_model_name, mfs_bool, topic_only, reweight, 
                reserve, sentence_only, sbert_name)
    
    evaluation_data, sent2ans, first_sense = load_data(filetype, mfs_bool)
    save_filename = gen_save_filename(sentence_only, mfs_bool, topic_only, reserve,
                                    trained_model_name, filetype, reweight, sbert_name)
                                    
    if mfs_bool:
        process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, 
                         sentence_only, sent, targetword, save_filename) 

    total_count, rank_score, top_one = 0, 0, 0
    for targetword, sentences in tqdm(evaluation_data.items()):
        for sentence in sentences:
            sent, ans = fetch_ans(filetype, sentence, sent2ans)
            if filetype in ['100', '200', '300'] and not ans:
                continue
                
            # if it is most frequent sense
            if mfs_bool:
                top_one = process_mfs_sense(first_sense, targetword, ans, top_one, mfs_bool, 
                                            sentence_only, sent, targetword, save_filename)
            else:
                results, input_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(sentence[0], 
                                                                                           'noun', targetword)
                if not results:
                    results, input_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(sentence[0],
                                                                                               'propn', targetword)
                senses = [line[0].replace('\n', '').strip() for line in results]
                
                if ans.endswith(senses[0]):
                    top_one += 1
                for idx, sense in enumerate(senses):
                    if ans.endswith(sense):
                        rank_score += 1 / (idx + 1)
                topics = list(token_scores.keys())
                write_data(mfs_bool, sentence_only, input_sent, targetword,
                       ans, senses, topics, save_filename)
            total_count += 1

    write_score(save_filename, rank_score, top_one, total_count)

if __name__ == '__main__':
    main()