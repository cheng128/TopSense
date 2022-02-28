from random import sample 

def load_data():
    filename = '../data/training_data/no_reserve/20-False-concat_masked_noun.tsv'
    with open(filename) as f:
        data = [line for line in f.readlines()]

    return data

def write_data(filename, data):
    with open(filename, 'w') as f: 
        for line in data:
            f.write(line)

def main():
    data = load_data()
    data = sample(data, len(data))
    edge = int(len(data) * 0.8)
    training_data = data[:edge]
    validation_data = data[edge:]
    write_data('../data/training_data/20-False-concat_train_noun.tsv', 
                training_data)
    write_data('../data/validation_data/20-False-concat_validate_noun.tsv', 
                validation_data)            

    return


if __name__ == '__main__':
    main()