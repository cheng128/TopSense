import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, default=0.0)
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-n', type=str, default=20)
    parser.add_argument('-pos', type=str, default='verb')
    args = parser.parse_args()
    threshold = args.t
    reserve = bool(args.r)
    number = args.n
    pos_tag = args.pos

    directory = 'reserve' if reserve else 'no_reserve'
    base_dir = f'../data/training_data/{pos_tag}'
    concat_file_name = f'{base_dir}/{directory}/{threshold}_{number}_{reserve}_concat.tsv'
    print('save as:', concat_file_name)

    with open(concat_file_name, 'w') as f:
        with open(f'{base_dir}/{threshold}_{reserve}_{pos_tag}_cambridge.tsv') as g:
            for line in g.readlines():
                f.write(line)

        with open(f'{base_dir}/{directory}/wiki_{number}_monosemous_{threshold}_{reserve}.tsv') as h:
            for line in h.readlines():
                f.write(line)
        
        with open(f'{base_dir}/{directory}/cam_all_monosemous_{threshold}_{reserve}.tsv') as d:
            for line in d.readlines():
                if len(line.split('\t')) == 3:
                    f.write(line)
                else:
                    print(line)


if __name__ == '__main__':
    main()