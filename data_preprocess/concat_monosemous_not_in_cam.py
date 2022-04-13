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
    
    filename = f'../data/training_data/{directory}/cam_wiki_mono_{num}_{reserve}_concat.tsv'
    print('save as:', filename)
    with open(filename, 'a') as g:
        concat_filename = f'../data/training_data/{directory}/0.6_remap_{num}_{reserve}_concat.tsv'
        print(concat_filename)
        with open(concat_filename) as h:
            for line in h.readlines():
                g.write(line)
#         with open(f'../data/training_data/{directory}/{reserve}_not_in_cam_masked_noun.tsv') as f:
#             for line in f.readlines():
#                 g.write(line)
        with open(f'../data/training_data/{directory}/{reserve}_monosemous_masked_noun.tsv') as f:
            for line in f.readlines():
                g.write(line)
                
        with open(f'../data/training_data/{directory}/{reserve}_cam_mono_masked_noun.tsv') as f:
            for line in f.readlines():
                g.write(line)
                
if __name__ == '__main__':
    main()