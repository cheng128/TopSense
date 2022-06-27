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
    pos_map = {'propn': 'noun', 'pron': 'noun', 'adverb': 'adv', 'adjective': 'adj'}
    
    pos_tag = pos_map[pos_tag]
    reconstruct = []
    topic_construct = []
    find = False

    for token in sent_list:
        # not found and the POS tag is match, we need to check this token
        token_pos = 'noun' if token['pos'] in ['PROPN', 'PRON'] else token['pos'].lower()
        if not find and pos_tag == token_pos.lower():
            if targetword.lower() == token['lemma'].lower():
                reconstruct, topic_construct = update_list(reserve, reconstruct, topic_construct,
                                                token, topic)
                find = True
                continue
        # when we are dealing with training data
        elif not find and targetword.lower() == token['text'].lower():
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