import json
import argparse
from transformers import BertTokenizerFast

def get_new_tokens(cased):
    with open('../data/BRT_data.json') as f:
        brt_data = json.load(f)

    new_tokens = set()
    for supergroup, categories in brt_data.items():
        for category, data in categories.items():
            category_name = category if cased else category.lower()
            topic_name = '[' + data['cat_num'] + ' ' + category_name + ']'
            new_tokens.add(topic_name)

    return list(new_tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help="cased", type=int)
    args = parser.parse_args()
    cased = bool(args.c)
    
    pretrain_name = 'bert-base-cased' if cased else 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(pretrain_name)

    new_tokens = get_new_tokens(cased)
    num_added_toks = tokenizer.add_tokens(new_tokens)
    tokenizer.save_pretrained(f'../topic_tokenizer_cased{cased}')

if __name__ == '__main__':
    main()