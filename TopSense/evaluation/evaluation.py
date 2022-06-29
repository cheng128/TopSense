import argparse
from tqdm import tqdm
from .evaluator_class import Evaluator
import sys
from ..disambiguator_class import Disambiguator 
from ..data_class import Data
from ..util import tokenize_processor

def main():
    parser = argparse.ArgumentParser()
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

    EVALUATOR = Evaluator(args)    
    EVALUATOR.print_info()
    
    evaluation_data, sent2ans, first_sense = EVALUATOR.load_data()
    save_filename = EVALUATOR.gen_save_filename()

    tokenizer_name = './TopSense/tokenizer_casedFalse'
    DATA = Data(args.sm, './TopSense/data')
    word2pos_defs, topic_embs, sense_examples_embs = DATA.load_data()
    sbert_model = DATA.load_sbert_model()
     
    DISAMBIGUATOR = Disambiguator(word2pos_defs, topic_embs, sense_examples_embs,
                                  sbert_model, args.m, tokenizer_name, args.r, args.s, 
                                  args.w, args.t)

    total_count, rank_score, top_one = 0, 0, 0
    for targetword, sentences in tqdm(evaluation_data.items()):
        for ans_key, sentence in sentences:
            ans = sent2ans.get(ans_key, '') 
            if not ans:
                continue
            if EVALUATOR.mfs_bool:
                top_one = EVALUATOR.process_mfs_sense(first_sense, targetword, ans, 
                                                      top_one, sentence, save_filename)
            else:
                tokens = tokenize_processor(sentence)
                results, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                            sentence, 
                                                                                            args.pos, 
                                                                                            targetword)
                if not results and args.pos == 'noun':
                    results, masked_sent, token_scores = DISAMBIGUATOR.predict_and_disambiguate(tokens,
                                                                                                sentence,
                                                                                                'propn', 
                                                                                                targetword)

                senses = [line[0].replace('\n', '').strip() for line in results]
                if ans.endswith(senses[0]):
                    top_one += 1
                for idx, sense in enumerate(senses):
                    if ans.endswith(sense):
                        rank_score += 1 / (idx + 1)
                topics = list(token_scores.keys())
                EVALUATOR.write_data(masked_sent, targetword, ans, senses, topics, save_filename)
            total_count += 1
            
    EVALUATOR.write_score(save_filename, rank_score, top_one, total_count)

if __name__ == '__main__':
    main()