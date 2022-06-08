import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=0.0)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-n', type=str, default=20)
    args = parser.parse_args()
    threshold = args.t
    reserve = bool(args.r)
    number = args.n

    directory = 'reserve' if reserve else 'no_reserve'
    concat_file_name = f'../data/training_data/{directory}/{threshold}_{number}_wiki_cam_monosemous_{reserve}_concat.tsv'
    print('save as:', concat_file_name)
    with open(concat_file_name, 'w') as f:

        with open(f'../data/training_data/{threshold}_monosemous_verb_{reserve}.tsv') as h:
            for line in h.readlines():
                f.write(line)
        with open(f'../data/training_data/{threshold}_{reserve}_verb_cambridge.tsv') as g:
            for line in g.readlines():
                f.write(line)
        with open(f'../data/training_data/{directory}/{threshold}_{number}_wikipedia_monosemous_verb_{reserve}.tsv') as d:
            for line in d.readlines():
                if len(line.split('\t')) == 3:
                    f.write(line)
                else:
                    print(line)


if __name__ == '__main__':
    main()