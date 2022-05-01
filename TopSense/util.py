import spacy

spacy_model = spacy.load("en_core_web_sm")

def tokenize_processor(sentence):

    doc = spacy_model(sentence)
    tokens = []
    for token in doc:
        text = token.text
        lemma = token.lemma_
        pos = token.pos_
        tokens.append({'text': text, 'lemma': lemma, 'pos': pos})
    return tokens

def update_list(reserve, reconstruct, topic_construct, token, topic):
    if reserve:
        reconstruct.append(token['text'] + ' [MASK]')
        if topic:
            topic_construct.append(f"{token['text']} {topic}")
    else:
        reconstruct.append('[MASK]')
        if topic:
            topic_construct.append(topic)
    return reconstruct, topic_construct

def gen_masked_sent(sent_list, pos_tag, targetword, reserve, topic=''):
    
    pos_tag = 'NOUN' if pos_tag in ['PROPN', 'PRON'] else pos_tag
    reconstruct = []
    topic_construct = []
    find = False

    for token in sent_list:
        candidates = [token['text'].lower(), token['lemma'].lower()]
        # not found and the POS tag is match, we need to check this token
        token_pos = 'NOUN' if token['pos'] in ['PROPN', 'PRON'] else token['pos']
        if not find and pos_tag.upper() == token_pos:
            if targetword.lower() in candidates:
                reconstruct, topic_construct = update_list(reserve, reconstruct, topic_construct,
                                                token, topic)
                find = True
                continue
        # when we are dealing with training data
        elif not find and targetword.lower() in candidates:
            reconstruct, topic_construct = update_list(reserve, reconstruct, topic_construct,
                                                        token, topic) 
            find = True
            continue
        reconstruct.append(token['text'])
        topic_construct.append(token['text'])

    masked_sent = ' '.join(reconstruct)
    topic_sent = ' '.join(topic_construct)
    masked_sent = '' if '[MASK]' not in masked_sent else masked_sent
    return masked_sent, topic_sent