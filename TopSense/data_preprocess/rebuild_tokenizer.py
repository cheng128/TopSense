import json
from collections import Counter
from transformers import pipeline


def load_data():
    
    with open('../data/all_tokenizer_rebuild.json') as f:
        tokenizer_rebuild = json.load(f)
    
    with open('../data/model.json') as f:
        model_list = json.load(f)
    
    with open('../data/BRT_data.json') as f:
        brt_data = json.load(f)
    
    return tokenizer_rebuild, model_list, brt_data

def load_model(model_name):
    tokenizer_name = "../tokenizer_casedFalse"
    model = pipeline('fill-mask',
           model=f'../model/{model_name}',
           tokenizer=tokenizer_name)
    return model

def pick_topic(dictionary):
    for key, value in dictionary.items():   
        max_item = sorted(value.items(), key=lambda x: x[1], reverse=True)[0][0]
        dictionary[int(key)] = max_item
    return dictionary

def retrieve_category_num(brt_data):
    category_num = {}
    for supergroup, categories in brt_data.items():
        for category, pos_data in categories.items():
            category_num[category] = pos_data['cat_num']
    return category_num

def add_num_to_topics(category_num, num2results):
    lower2upper = {}
    new_map = {}
    for key, value in category_num.items():
        lower2upper[key.lower()] = key
        new_map[key.lower()] = value

    topic_results = {}
    for num, category in num2results.items():
        if category and category in new_map:
            topic_results[num] = '[' + new_map[category] + ' ' + lower2upper[category] + ']'
    return topic_results

def main():
    tokenizer_rebuild, model_list, brt_data = load_data()
    
    tokenizer_rebuild_int = pick_topic(tokenizer_rebuild)
    
    num2results = defaultdict(list)
    for model_name, evaluation_file in model_list.items():
        MLM = load_model(model_name)
        with open(f'../evaluation/results/{evaluation_file}') as f:
            for line in tqdm(f.readlines()):
                split_data = line.split('\t')
                masked_sent = split_data[0]
                if '[MASK]' in masked_sent:
                    for idx, result in enumerate(MLM(masked_sent)):
                        if result['token_str'].startswith('['):
                            num2results[result['token']].append(split_data[idx+2])
    
    for num, topics in num2results.items():
        num2results[num] = Counter(topics)
    
    
    num2results = pick_topic(num2results)
    category_num = retrieve_category_num(brt_data)
    topic_results = add_num_to_topics(category_num)
            
    tokenizer_rebuild_int.update(topic_results, num2results)
    
    with open('../data/all_tokenizer_map.json', 'w') as f:
        f.write(json.dumps(tokenizer_rebuild_int))
        
if __name__ == '__main__':
    main()