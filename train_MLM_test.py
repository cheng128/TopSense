import torch
import json
import os
import argparse
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, AdamW

class TopicDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def load_preprocessed(filename):
    topic_sent = []
    masked_sent = []

    print('read file:', filename)
    with open('./data/' + filename) as f:
        for line in f.readlines():
            pieces = line.split('\t')
            topic_sent.append(pieces[1])
            masked_sent.append(pieces[2])
    return masked_sent, topic_sent

def gen_dataloader(masked_sent, topic_sent, tokenizer):
    max_length = 64
    batch_size = 64
    inputs = tokenizer(masked_sent,
                       return_tensors='pt',
                       max_length=max_length,
                       truncation=True,
                       padding='max_length')
    
    inputs['labels'] = tokenizer(topic_sent,
                                 return_tensors='pt',
                                 max_length=max_length,
                                 truncation=True,
                                 padding='max_length')["input_ids"]

    dataset = TopicDataset(inputs)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True)
    return dataloader

def creat_directory(args):
    map_dict = {'test': 'test', 'wiki': 'wikipedia', 
                'concat': 'concat', 'gbook': 'gbook'}
    if args.save:
        dir_path = f'./model/{map_dict[args.g]}/{args.n}_{args.r}_{args.e}epochs_test'
        try:
            os.mkdir(dir_path)
        except:
            dir_path += '_' + str(1)
            os.mkdir(dir_path)
        print('model will be saved in: ', dir_path)
    return dir_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="#epochs", type=int)
    parser.add_argument('-t', type=str) # training data filename
    parser.add_argument('-v', type=str) # validation data filename
    parser.add_argument('-g', type=str) # directoy
    parser.add_argument('--save', action="store_true")
    parser.add_argument('-n', type=str)
    parser.add_argument('-r', type=int)
    args = parser.parse_args()
    
    reserve = bool(args.r)
    training_filename = '/training_data/' + args.t
    validate_filename = '/validation_data/' + args.v

    dir_path = creat_directory(args) 
        
    tokenizer_name = 'remap_tokenizer'
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print("use tokenizer:", tokenizer_name)
    
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # must implement
    model.resize_token_embeddings(len(tokenizer))
    
    train_masked_sent, train_topic_sent = load_preprocessed(training_filename)
    validate_masked_sent, validate_topic_sent = load_preprocessed(validate_filename)
    print('after load data')
    
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    print('device: ', device)

    train_dataloader = gen_dataloader(train_masked_sent, train_topic_sent, tokenizer)
    validate_dataloader = gen_dataloader(validate_masked_sent, validate_topic_sent, tokenizer)
    
    model.to(device)

    optim = AdamW(model.parameters(), lr=5e-6)
    train_loss_record = []
    validate_loss_record = []
    for epoch in range(args.e):
        train_loop = tqdm(train_dataloader, leave=True)
        model.train()
        for batch in train_loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            train_loop.set_description(f'Epoch {epoch}')
            train_loop.set_postfix(loss=loss.item()) 
        train_loss_record.append([str(epoch), str(float(loss))])
        
        model.eval()
        validate_loop = tqdm(validate_dataloader, leave=True)
        for batch in validate_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0] 
    
            validate_loop.set_description(f'Epoch {epoch}')
            validate_loop.set_postfix(loss=loss.item()) 
        validate_loss_record.append([str(epoch), str(float(loss))])


    if args.save:
        model.save_pretrained(dir_path)
        with open(f'{dir_path}/spec.txt', 'w') as f:
            f.write(f'python train_MLM.py -e {args.e} -g {args.g} -t {args.t} -v {args.v} -n {args.n} -r {args.r}\n')
            for train_loss, validate_loss in zip(train_loss_record, validate_loss_record):
                f.write('train_loss: ' + '\t'.join(train_loss) + '\n')
                f.write('validate_loss: ' + '\t'.join(validate_loss) + '\n')
            
            
if __name__ == '__main__':
    main()