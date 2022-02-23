import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str)
    parser.add_argument('-r', type=int)
    args = parser.parse_args()
    
    reserve = bool(args.r)
    directory = 'reserve' if reserve else 'no_reserve'
    num = args.n

    with open(f'../data/training_data/{directory}/{num}-{reserve}-concat_masked_noun.tsv', 'a') as g:
        with open(f'../data/training_data/{reserve}_remap_brt_masked_noun.tsv') as f:
            for line in f.readlines():
                g.write(line)
        with open(f'../data/training_data/{directory}/{num}-{reserve}-t5-xl_wiki_masked.tsv') as h:
            for line in h.readlines():
                g.write(line)
        
if __name__ == '__main__':
    main()