import json
from collections import defaultdict

def main():
    with open('../data/cambridge.sense.000.jsonl') as f:
        cambridge = [json.loads(line) for line in f.readlines()]

    words2def = defaultdict(dict)

    for line in cambridge:
        headword = line['headword']
        definition = line['en_def']
        pos = line['pos']
        if pos not in words2def[headword]: 
            words2def[headword][pos] = []

        words2def[headword][pos].append(definition)

    with open('../data/word2pos_defs.json', 'w') as f:
        f.write(json.dumps(words2def))

if __name__ == '__main__':
    main()