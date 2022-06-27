import json
import random
import argparse
from tqdm import tqdm

def load_data():
    with open('../data/wiki/all_wikipedia_monosemous.json') as f:
        monosemous_data = json.load(f)
    return monosemous_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=20)
    args = parser.parse_args()
    sample_number = args.n
    
    monosemous_data = load_data()
    
    for headword, value in tqdm(monosemous_data.items()):
        for pos, data in value.items():    
            examples = data['examples']
            if len(examples) > sample_number:
                examples = random.sample(examples, sample_number)
            monosemous_data[headword][pos]['examples'] = examples
            
    with open(f'../data/wiki/{sample_number}_wikipedia_monosemous.json', 'w') as f:
        f.write(json.dumps(monosemous_data))
                
    
if __name__ == '__main__':
    main()