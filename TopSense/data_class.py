import json
import pickle
import spacy
from sentence_transformers import SentenceTransformer

class Data:
    
    def __init__(self, model_name, data_directory, device='gpu'):
        self.model_name = model_name
        self.data_directory = data_directory
        self.device = device

    def load_emb(self, filename):

        print("embedding file:", filename)    
        with open(filename, 'rb') as f:
            emb_map = pickle.load(f)

        return emb_map

    def load_data(self):
        with open(f'{self.data_directory}/word2pos_defs.json') as f:
            word2pos_defs = json.loads(f.read())
        
        topic_emb_filename = f'{self.data_directory}/embeddings/{self.model_name}_topic_embs_{self.device}.pickle'
        topic_embs = self.load_emb(topic_emb_filename)
        
        sense_embs_filename = f'{self.data_directory}/embeddings/{self.model_name}_sense_embs_{self.device}.pickle'
        sense_examples_embs = self.load_emb(sense_embs_filename)

        return word2pos_defs, topic_embs, sense_examples_embs

    def load_sbert_model(self):
        sbert_model = SentenceTransformer(self.model_name)
        return sbert_model

    def load_spacy_model(self):
        spacy_model = spacy.load("en_core_web_sm")
        return spacy_model
