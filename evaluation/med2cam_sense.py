import json
import pickle
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-roberta-large-v1')

def load_data():

    with open('../data/words2defs.json') as f:
        word2defs = json.loads(f.read())
    
    with open('../data/definition_emb.pickle', 'rb') as f:
        def2emb = pickle.load(f)
        
    with open('./data/MED.json') as f:
        med_data = json.loads(f.read())
    
    return word2defs, def2emb, med_data
    
def build_med_map(med_data):         
    """ map med word to sense and examples(only noun)
    """
    med_map = defaultdict(list)
    for word, pos in med_data.items():
        if 'noun' in pos:
            for sense in pos['noun']['SENSE']:
                med_map[word].append({"sense": sense[0],
                                      "examples": [sentence[0] for sentence in sense[-1]]})
    return med_map
            
def main():
    word2defs, def2emb, med_data = load_data()
    med_map = build_med_map(med_data)

    med2cam_sense = defaultdict(list)
    
    for word in tqdm(med_data.keys()):
        if word in word2defs:
            for med_sense in med_map[word]:
                max_sim = float('-inf')
                proper_sense = ''
                for cam_sense in word2defs[word]:
                    if med_sense['sense']:
                        sentences = [med_sense['sense']]
                        embeddings = model.encode(sentences, convert_to_tensor=True)
                        cosine_scores = util.cos_sim(embeddings, def2emb[cam_sense])

                        avg_sim = 0
                        for j in range(0, len(def2emb[cam_sense])):
                            avg_sim += cosine_scores[0][j]
                        avg_sim = avg_sim / len(def2emb[cam_sense])
                        if avg_sim > max_sim:
                            max_sim = avg_sim
                            proper_sense = cam_sense
                if proper_sense:
                    med2cam_sense[word].append({'med_sense': med_sense['sense'], 
                                                'cam_sense': proper_sense,
                                                'examples': med_sense['examples'],
                                                'avg_score': float(max_sim)})

    with open('./data/med2cam_sense.noun.json', 'w') as f:
        f.write(json.dumps(med2cam_sense))
        
if __name__ == '__main__':
    main()