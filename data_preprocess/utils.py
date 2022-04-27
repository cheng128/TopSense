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

def handle_examples(headword, sent_en, topic, reserve=False, pos_tag="NOUN"):
    reconstruct = []
    topic_construct = []
    doc = nlp(sent_en)
    
    word = ''
    count = 0
    for token in doc:
        find = False
        if token.text == headword and pos_tag.upper() == token.pos_:
            word = token.text
            find = True
        elif token.lemma_ == headword and pos_tag.upper() == token.pos_:
            word = token.lemma_
            find = True

        if find and count == 0:
            if reserve:
                reconstruct.append(word + ' [MASK]')
                topic_construct.append(word + f' {topic}')
            else:
                reconstruct.append('[MASK]')
                topic_construct.append(f'{topic}')
            count += 1
            continue
        
        reconstruct.append(token.text)
        topic_construct.append(token.text)
            
    masked_sent = ' '.join(reconstruct)
    topic_sent = ' '.join(topic_construct)
    
    if '[MASK]' not in masked_sent:
        topic_sent, masked_sent = '', ''
        
    return topic_sent, masked_sent 

def handle_exceptions(headword, doc, reconstruct, topic_construct, topic, reserve):
    
    target_idx = 0
    find = False
    
    for idx, token in enumerate(doc):
        if token.text.startswith(headword) and len(token.text) - len(headword) < 3:
            target_idx = idx
            find = True
    
    if find and reserve:
        reconstruct[target_idx] = f'{headword} [MASK]'
        topic_construct[target_idx] = f'{headword} {topic}'
    elif find:
        reconstruct[target_idx] = '[MASK]'
        topic_construct[target_idx] = topic

    return reconstruct, topic_construct

def one_word_examples(headword, sent_en, topic, reserve=False):
    reconstruct = []
    topic_construct = []
    doc = nlp(sent_en)
    
    word = ''
    count = 0
    for token in doc:
        find = False
        if token.text.lower() == headword.lower():
            word = token.text
            find = True
        elif token.lemma_ == headword.lower():
            word = token.lemma_
            find = True
            
        if find and count == 0:
            if reserve:
                reconstruct.append(f'{word} [MASK]')
                topic_construct.append(f'{word} {topic}')
            else:
                reconstruct.append('[MASK]')
                topic_construct.append(f'{topic}')
            count += 1
            continue

        reconstruct.append(token.text)
        topic_construct.append(token.text)

    if '[MASK]' not in reconstruct:
        reconstruct, topic_construct = handle_exceptions(headword,
                                                         doc,
                                                         reconstruct,
                                                         topic_construct, 
                                                         topic, 
                                                         reserve)
    masked_sent = ' '.join(reconstruct)
    topic_sent = ' '.join(topic_construct)

    if '[MASK]' not in masked_sent:
        masked_sent = ''

    return topic_sent, masked_sent 

def multi_words_examples(headword, sent_en, topic, reserve=False):
    word_length = len(headword.split())
    sentence_split = sent_en.split()
    topic_construct = sentence_split[:]
    sent_ngrams = [' '.join(gram).lower()  for gram in ngrams(sentence_split, word_length)]
    target_idx = 0
    find = False
    
    for idx, ngram in enumerate(sent_ngrams):
        if ngram.startswith(headword.lower()):
            target_idx = idx
            find = True
            
    pending = [''] * (word_length - 1) 
    if find and reserve:
        sentence_split[target_idx:target_idx+word_length] = [f'{headword} [MASK]'] + pending 
        topic_construct[target_idx:target_idx+word_length] = [f'{headword} {topic}']  + pending 
    elif find:
        sentence_split[target_idx:target_idx+word_length] = [f'[MASK]']  + pending 
        topic_construct[target_idx:target_idx+word_length] = [f'{topic}'] + pending

    if find and pending:
        masked_sent = ' '.join(sentence_split).replace(''.join(pending), '')
        topic_sent = ' '.join(topic_construct).replace(''.join(pending), '')
    elif find:
        masked_sent = ' '.join(sentence_split)
        topic_sent = ' '.join(topic_construct)
    else:
        masked_sent, topic_sent = '', ''
    
    return topic_sent, masked_sent