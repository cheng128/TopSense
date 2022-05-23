import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=0.0)
    parser.add_argument('-r', type=int, default=0)
    args = parser.parse_args()
    threshold = args.t
    reserve = bool(args.r)

    directory = 'reserve' if reserve else 'no_reserve'
    concat_file_name = f'../data/training_data/{directory}/{threshold}_{reserve}_verb_noun_wiki_concat.tsv'
    print('save as:', concat_file_name)
    with open(concat_file_name, 'w') as f:
        # with open(f'../data/training_data/{threshold}_{reserve}_noun_cambridge.tsv') as h:
        #     for line in h.readlines():
        #         f.write(line)
        with open(f'../data/training_data/{directory}/{threshold}_remap_20_{reserve}_concat.tsv') as h:
            for line in h.readlines():
                f.write(line)
        with open(f'../data/training_data/{threshold}_{reserve}_verb_cambridge.tsv') as g:
            for line in g.readlines():
                f.write(line)



if __name__ == '__main__':
    main()