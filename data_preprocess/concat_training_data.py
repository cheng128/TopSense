import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str)
    parser.add_argument('-r', type=int)
    parser.add_argument('-v', type=str)
    args = parser.parse_args()
    
    reserve = bool(args.r)
    directory = 'reserve' if reserve else 'no_reserve'
    num = args.n
    version = args.v
    
    filename = f'../data/training_data/{directory}/0.6_remap_{num}_{reserve}_concat.tsv'
    print('save as:', filename)
    with open(filename, 'a') as g:
        with open(f'../data/training_data/0.6_cross_ref_reserve{reserve}_cambridge.tsv') as f:
            for line in f.readlines():
                g.write(line)
        with open(f'../data/training_data/{directory}/0.6_reremap_{num}_{version}_{reserve}.tsv') as h:
            for line in h.readlines():
                g.write(line)
#         with open(f'../data/training_data/{directory}/{num}_gbook_{reserve}_noun.tsv') as f:
#             for line in f.readlines():
#                 g.write(line)
        
if __name__ == '__main__':
    main()