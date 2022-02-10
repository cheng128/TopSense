from transformers import pipeline

nlp = spacy.load("en_core_web_sm")
model = pipeline('fill-mask',
                 model=f"../model/wiki_similarity/all_10epochs",
                 tokenizer="../topic_tokenizer")


