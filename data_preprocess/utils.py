import nltk
import spacy
import string

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
punct = set(string.punctuation) 
nlp = spacy.load("en_core_web_sm")

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word,'n')
    return lemma    

def clean_text(sentence):
    tokenized = nltk.word_tokenize(sentence)
    clear_punct = [s for s in tokenized if s not in punct]
    lemmatized = [lemmatize(s) for s in clear_punct] 
    return lemmatized

def handle_examples(headword, sent_en, topic, reserve=False):
    reconstruct = []
    topic_construct = []
    doc = nlp(sent_en)
    
    word = ''
    count = 0
    for token in doc:
        find = False
        if token.text == headword:
            word = token.text
            find = True
        elif token.lemma_ == headword:
            word = token.lemma_
            find = True

        if find and count == 0:
            if reserve:
                reconstruct.append(word + '[MASK]')
                topic_construct.append(word + f'{topic}')
            else:
                reconstruct.append('[MASK]')
                topic_construct.append(f'{topic}')
            count += 1
            continue
        
        reconstruct.append(token.text)
        topic_construct.append(token.text)
            
    masked_sent = ' '.join(reconstruct)
    topic_sent = ' '.join(topic_construct)
    return topic_sent, masked_sent 

