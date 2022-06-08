import os
import json
import torch
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForMaskedLM, AdamW

class TopicDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def make_dir(args):
    dir_path = f'./model/{args.g}/{args.m}_{args.n}_{bool(args.r)}_{args.e}epochs_{args.lr}'
    try:
        os.mkdir(dir_path)
    except:
        dir_path += '_' + str(1)
        os.mkdir(dir_path)
    print('model will be saved in: ', dir_path)
    return dir_path

def load_model_tokenizer(pretrain):
    tokenizer_name = 'tokenizer_casedFalse'  
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print("use tokenizer:", tokenizer_name)
    print("pretrained model:", pretrain)

    MLM = BertForMaskedLM.from_pretrained(pretrain)
    
    # must implement
    MLM.resize_token_embeddings(len(tokenizer))
    return MLM, tokenizer

def load_preprocessed_file(filename):
    topic_sent, masked_sent = [], []

    print('read file:', filename)
    with open('./data/' + filename) as f:
        for line in f.readlines():
            pieces = line.split('\t')
            topic_sent.append(pieces[1])
            masked_sent.append(pieces[2])
    return topic_sent, masked_sent

def gen_data_loader(masked_sent, topic_sent, tokenizer):
    inputs = tokenizer(masked_sent,
                       return_tensors='pt',
                       max_length=64,
                       truncation=True,
                       padding='max_length')
    
    inputs['labels'] = tokenizer(topic_sent,
                                 return_tensors='pt',
                                 max_length=64,
                                 truncation=True,
                                 padding='max_length')["input_ids"]

    dataset = TopicDataset(inputs)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

def gen_train_test_data(masked_sent, topic_sent, tokenizer):
    train_masked, test_masked, train_topic, test_topic = train_test_split(masked_sent,
                                                                          topic_sent, 
                                                                          test_size=0.1, 
                                                                          random_state=42)
    train_dataloader = gen_data_loader(train_masked, train_topic, tokenizer)
    test_dataloader = gen_data_loader(test_masked, test_topic, tokenizer)
    return train_dataloader, test_dataloader

def write_spec(temp_dir_path, epochs, args, filename, train_loss_record, test_loss_record):
    with open(f'{temp_dir_path}/spec.txt', 'w') as f:
        f.write(f'python train_MLM_with_validation.py -e {epochs} -g {args.g}
                    -f {filename} -n {args.n} -r {args.r} -m {args.m} -lr {args.lr}\n')
        f.write(f'tokenizer: tokenizer_casedFalse \n')
        f.write(f'pretrained model: {args.pre}\n')
        for loss, validate in zip(train_loss_record, test_loss_record):
            f.write('\t'.join(loss) + '\t' + validate + '\n') 

def train_MLM(filename, lr, epochs, dir_path, pretrain, args):

    MLM, tokenizer = load_model_tokenizer(pretrain)
    topic_sent, masked_sent = load_preprocessed_file(filename)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MLM.to(device)
    MLM.train()

    optim = AdamW(MLM.parameters(), lr=lr)
    train_loss_record, test_loss_record = [], []

    for epoch in range(epochs):
        train_dataloader, test_dataloader = gen_train_test_data(masked_sent, topic_sent, tokenizer)
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = MLM(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        train_loss_record.append([str(epoch), str(float(loss))])

        # validation
        with torch.no_grad():
            loop = tqdm(test_dataloader, leave=True)

            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = MLM(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0] 
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item()) 
            test_loss_record.append(str(float(loss)))
        
        temp_dir_path = dir_path.replace(f'{epochs}epochs', f'{epoch+1}epochs')
        MLM.save_pretrained(temp_dir_path)
        
        write_spec(temp_dir_path, epochs, args, filename, train_loss_record, test_loss_record)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="#epochs", type=int)
    parser.add_argument('-f', type=str) # preprocess file name
    parser.add_argument('-g', type=str) # directoy
    parser.add_argument('-n', type=str, default='all')
    parser.add_argument('-r', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-m', type=str)
    parser.add_argument('-pre', type=str, default='bert-base-uncased')
    args = parser.parse_args()

    filename = args.f
    lr = args.lr
    epochs = args.e
    pretrain = args.pre

    dir_path = make_dir(args)
        
    train_MLM(filename, lr, epochs, dir_path, pretrain, args)
            
if __name__ == '__main__':
    main()